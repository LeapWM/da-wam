"""Prepared measured geometry with exact-reference fallbacks at boundaries.

Preserves recorded_geometry.targets semantics. Cached hulls describe actual
input frames; no actor state is copied into a missing frame. Queries reuse a
reference hull only when a well-conditioned affine fit reconstructs their
corners within 1e-9 m; arbitrary queries use the general hull implementation.
Potentially ambiguous contact thresholds also use that general implementation.
"""
from dataclasses import dataclass
import hashlib
from types import SimpleNamespace

import numpy as np
from scipy.spatial import QhullError

from recorded_geometry import GeometryTargets, _ConvexBox, intersects


TOLERANCE=1e-6
AFFINE_RESIDUAL=1e-9
BOUNDARY_GUARD=1e-7


@dataclass(frozen=True)
class HullGeometry:
    # Plain arrays/scalars: a prepared scene can cross worker processes or be
    # saved without attempting to pickle a SciPy Qhull runtime object.
    points: np.ndarray
    origin: np.ndarray
    relative: np.ndarray
    normals: np.ndarray
    edges: np.ndarray
    low: np.ndarray
    high: np.ndarray
    volume: float


@dataclass(frozen=True)
class PreparedFrame:
    present: bool
    reference: object
    reference_pinv: object
    objects: tuple
    invalid_objects: int
    object_low: np.ndarray
    object_high: np.ndarray


@dataclass(frozen=True)
class PreparedScene:
    frames: tuple
    times_s: np.ndarray
    fingerprint: str
    constructed_hulls: int
    source: str = 'RECORDED_ANNOTATION'


@dataclass(frozen=True)
class PreparedResult:
    labels: GeometryTargets
    statistics: dict


def frozen_array(a):
    a=np.array(a,copy=True);a.setflags(write=False);return a


def prepare(reference_corners, observations, *, times_s, frame_present):
    reference=np.asarray(reference_corners,dtype=np.float64);times=np.asarray(times_s,dtype=float)
    present=np.asarray(frame_present);t=len(observations)
    if reference.shape!=(t,8,3) or not t:raise ValueError('Expected reference [T,8,3]')
    if times.shape!=(t,) or not np.isfinite(times).all() or (times<0).any() or (np.diff(times)<=0).any():
        raise ValueError('Expected actual increasing nonnegative times')
    if present.shape!=(t,) or present.dtype!=bool:raise ValueError('Expected boolean frame presence')
    digest=hashlib.sha256(b'prepared_recorded_geometry_v1');digest.update(reference.tobytes());digest.update(times.tobytes());digest.update(present.tobytes())
    memo={};frames=[];count=0
    def hull(vertices):
        nonlocal count
        points=np.asarray(vertices,dtype=np.float64)
        if points.shape!=(8,3) or not np.isfinite(points).all():return None
        key=points.tobytes()
        if key not in memo:
            try:
                box=_ConvexBox(points.copy())
                fields={attr:frozen_array(getattr(box,attr)) for attr in ('points','origin','relative','normals','edges','low','high')}
                memo[key]=HullGeometry(**fields,volume=float(box.hull.volume));count+=1
            except (ValueError,QhullError):memo[key]=None
        return memo[key]
    for ti,objects in enumerate(observations):
        if any(o.source!='RECORDED_ANNOTATION' for o in objects):raise ValueError('No CV, interpolated future or unknown sources')
        ids=[o.actor_id for o in objects]
        if any(not isinstance(x,str) or not x for x in ids) or len(set(ids))!=len(ids):raise ValueError('Invalid/duplicate actor IDs')
        if not present[ti] and objects:raise ValueError('Absent frame cannot contain objects')
        digest.update(str(len(objects)).encode()+b':')
        usable=[];invalid=0
        for obj in objects:
            identity=obj.actor_id.encode();digest.update(len(identity).to_bytes(8,'little')+identity)
            raw=np.asarray(obj.vertices,dtype=np.float64)
            digest.update(bytes([bool(obj.geometry_valid)]));digest.update(str(raw.shape).encode()+b':'+raw.tobytes())
            box=hull(obj.vertices) if obj.geometry_valid else None
            if box is None:invalid+=1
            else:usable.append((obj.actor_id,box))
        ref=hull(reference[ti]) if present[ti] else None
        pinv=frozen_array(np.linalg.pinv(ref.relative)) if ref is not None else None
        frames.append(PreparedFrame(bool(present[ti]),ref,pinv,tuple(usable),invalid,
            frozen_array(np.asarray([b.low for _,b in usable]).reshape(-1,3)),
            frozen_array(np.asarray([b.high for _,b in usable]).reshape(-1,3))))
    return PreparedScene(tuple(frames),frozen_array(times),digest.hexdigest(),count)


def narrow_gap(a,b):
    crosses=np.cross(a.edges[:,None,:],b.edges[None,:,:]).reshape(-1,3)
    normals=np.concatenate((a.normals,b.normals,crosses));lengths=np.linalg.norm(normals,axis=-1)
    normals=normals[lengths>1e-10]/lengths[lengths>1e-10,None]
    pa=a.relative@normals.T;pb=(b.relative+(b.origin-a.origin))@normals.T
    return float(np.maximum(pa.min(0)-pb.max(0),pb.min(0)-pa.max(0)).max())


def score(candidate_corners, scene, *, candidate_geometry_valid=None):
    corners=np.asarray(candidate_corners,dtype=np.float64)
    if corners.ndim!=4 or corners.shape[2:]!=(8,3) or min(corners.shape[:2])<=0:raise ValueError('Expected nonempty [P,T,8,3] queries')
    p,t=corners.shape[:2]
    if not isinstance(scene,PreparedScene) or len(scene.frames)!=t:raise ValueError('Prepared scene horizon mismatch')
    if scene.source!='RECORDED_ANNOTATION':raise ValueError('Prepared scene must contain recorded annotation geometry')
    mask=np.ones((p,t),dtype=bool) if candidate_geometry_valid is None else np.asarray(candidate_geometry_valid)
    if mask.shape!=(p,t) or mask.dtype!=bool:raise ValueError('Expected boolean candidate geometry validity')
    value=np.full((p,t),np.nan);valid=np.zeros((p,t),dtype=bool);witness=np.full((p,t),'',dtype=object)
    invalid=np.asarray([f.invalid_objects for f in scene.frames])
    stats=dict(affine_query_hulls=0,general_query_hulls=0,invalid_query_hulls=0,
        broadphase_pairs=0,broadphase_survivors=0,narrowphase_checks=0,boundary_fallback_checks=0)
    for ti,frame in enumerate(scene.frames):
        if not frame.present:continue
        indices=np.flatnonzero(mask[:,ti]&np.isfinite(corners[:,ti]).all(axis=(1,2)))
        if not len(indices):continue
        points=corners[indices,ti]
        with np.errstate(over='ignore',invalid='ignore'):
            centers=points.mean(axis=1);relative=points-centers[:,None,:]
        finite=np.isfinite(centers).all(axis=1)&np.isfinite(relative).all(axis=(1,2))
        stats['invalid_query_hulls']+=int((~finite).sum())
        indices=indices[finite];points=points[finite];centers=centers[finite];relative=relative[finite]
        if not len(indices):continue
        low=points.min(axis=1);high=points.max(axis=1)
        fitted=np.zeros(len(indices),dtype=bool);linear=None
        if frame.reference is not None:
            linear=np.einsum('ij,pjk->pik',frame.reference_pinv,relative)
            rebuilt=np.einsum('ij,pjk->pik',frame.reference.relative,linear)
            residual=np.abs(rebuilt-relative).max(axis=(1,2))
            # Very large/small or ill-conditioned transforms retain the
            # general Qhull behavior instead of changing numerical validity.
            with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
                determinant=np.linalg.det(linear);volume=abs(determinant)*frame.reference.volume
                condition=np.full(len(indices),np.inf)
                stable=np.isfinite(linear).all(axis=(1,2))&(np.abs(linear).max(axis=(1,2))<1e6)
                condition[stable]=np.linalg.cond(linear[stable])
            fitted=(residual<=AFFINE_RESIDUAL)&np.isfinite(volume)&(volume>1e-10)&(condition<1e6)
        near=~(np.any(low[:,None,:]>frame.object_high[None,:,:]+TOLERANCE,axis=-1)
            |np.any(frame.object_low[None,:,:]>high[:,None,:]+TOLERANCE,axis=-1))
        stats['broadphase_pairs']+=near.size;stats['broadphase_survivors']+=int(near.sum())
        for ii,pi in enumerate(indices):
            query=None;general=None
            if fitted[ii]:stats['affine_query_hulls']+=1
            else:
                try:general=_ConvexBox(points[ii]);query=general;stats['general_query_hulls']+=1
                except (ValueError,QhullError):stats['invalid_query_hulls']+=1;continue
            hit=None
            for oi in np.flatnonzero(near[ii]):
                actor,other=frame.objects[oi]
                if query is None:
                    mat=linear[ii];inverse=np.linalg.inv(mat)
                    query=SimpleNamespace(points=points[ii],origin=centers[ii],relative=relative[ii],
                        normals=frame.reference.normals@inverse.T,edges=frame.reference.edges@mat,
                        low=low[ii],high=high[ii])
                stats['narrowphase_checks']+=1
                gap=narrow_gap(query,other)
                if abs(gap-TOLERANCE)<=BOUNDARY_GUARD:
                    if general is None:general=_ConvexBox(points[ii]);stats['general_query_hulls']+=1
                    overlap=intersects(general,other);stats['boundary_fallback_checks']+=1
                else:overlap=gap<=TOLERANCE
                if overlap:hit=actor;break
            if hit is not None:value[pi,ti]=0.;valid[pi,ti]=True;witness[pi,ti]=hit
            elif invalid[ti]==0:value[pi,ti]=1.;valid[pi,ti]=True
    failed=valid&(value==0);any_failed=failed.any(axis=1);all_passed=(valid&(value==1)).all(axis=1)
    total_valid=any_failed|all_passed;total_value=np.where(any_failed,0.,np.where(all_passed,1.,np.nan))
    first=np.full(p,np.nan);first_valid=np.zeros(p,dtype=bool)
    for pi in range(p):
        if any_failed[pi]:
            ti=int(np.flatnonzero(failed[pi])[0])
            if valid[pi,:ti+1].all():first[pi]=scene.times_s[ti];first_valid[pi]=True
        elif all_passed[pi]:first[pi]=np.inf;first_valid[pi]=True
    labels=GeometryTargets(total_value,total_valid,value,valid,first,first_valid,witness,invalid)
    return PreparedResult(labels,stats)
