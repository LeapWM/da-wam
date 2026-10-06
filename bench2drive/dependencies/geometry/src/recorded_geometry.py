"""Safety-component targets from measured geometry, with no future predictor.

The metric is clearance against the explicitly supplied recorded annotation
boxes at the supplied times. It is NOT a world-complete safety certificate or
a physics-mesh / closed-loop success target. Callers supply candidate geometry;
this module does not infer candidate terrain, forecast actors, or create plans.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True)
class ObservedBox:
    actor_id: str
    vertices: np.ndarray             # exact recorded world corners [8,3]
    geometry_valid: bool = True      # valid for ANNOTATION-box metric
    source: str = 'RECORDED_ANNOTATION'


@dataclass(frozen=True)
class GeometryTargets:
    value: np.ndarray               # [P], 1 clear / 0 overlap / NaN unknown
    valid: np.ndarray               # [P]
    time_value: np.ndarray          # [P,T]
    time_valid: np.ndarray          # [P,T]
    first_overlap_s: np.ndarray     # inf for valid all-clear; NaN unknown
    first_overlap_valid: np.ndarray # earliest hit requires known-clear prefix
    witness_actor_id: np.ndarray   # [P,T]; id of measured conflicting object
    invalid_object_counts: np.ndarray # [T]
    definition: str = 'recorded_annotation_box_clear'
    source: str = 'RECORDED_ANNOTATION'


class _ConvexBox:
    def __init__(self, vertices):
        self.points=np.asarray(vertices,dtype=np.float64)
        if self.points.shape!=(8,3) or not np.isfinite(self.points).all():
            raise ValueError('Expected finite eight-corner world geometry')
        self.origin=self.points.mean(axis=0)
        self.relative=self.points-self.origin
        self.hull=ConvexHull(self.relative)
        if self.hull.volume<=1e-10:
            raise ValueError('Degenerate geometry')
        self.normals=self.hull.equations[:,:3]
        pairs=set()
        for face in self.hull.simplices:
            for i in range(3):
                for j in range(i+1,3):pairs.add(tuple(sorted((int(face[i]),int(face[j])))))
        self.edges=np.asarray([self.relative[j]-self.relative[i] for i,j in sorted(pairs)])
        self.low=self.points.min(axis=0);self.high=self.points.max(axis=0)


def intersects(a, b, tolerance_m=1e-6):
    """Convex polyhedron SAT, using original corners without fitting an OBB."""
    if np.any(a.low>b.high+tolerance_m) or np.any(b.low>a.high+tolerance_m):
        return False
    crosses=np.cross(a.edges[:,None,:],b.edges[None,:,:]).reshape(-1,3)
    normals=np.concatenate((a.normals,b.normals,crosses))
    lengths=np.linalg.norm(normals,axis=-1)
    normals=normals[lengths>1e-10]/lengths[lengths>1e-10,None]
    pa=a.relative@normals.T
    pb=(b.relative+(b.origin-a.origin))@normals.T
    gap=np.maximum(pa.min(0)-pb.max(0),pb.min(0)-pa.max(0))
    return bool(gap.max()<=tolerance_m)


def targets(candidate_corners, observations: Sequence[Sequence[ObservedBox]], *,
            times_s, frame_present, candidate_geometry_valid: Optional[np.ndarray]=None):
    """Compute labels for an explicitly scoped list of observed object boxes.

    Unknown object geometry blocks a clear target but cannot erase a verified
    overlap with another valid recorded object. An absent object is NOT filled
    forward/backward. Clearing an empty *recorded list* passes only this metric.
    A missing annotation frame never passes. First-overlap time is supervised
    only when its entire preceding sampled interval is known.

    P is generic for diagnostic queries; production callers must retain their
    own original32 contract. T must contain actual annotation times. No actor
    interpolation, candidate interpolation or ego extrapolation occurs here.
    """
    corners=np.asarray(candidate_corners,dtype=np.float64)
    if corners.ndim!=4 or corners.shape[2:]!=(8,3) or min(corners.shape[:2])<=0:
        raise ValueError('Expected nonempty [P,T,8,3] candidate corners')
    p,t=corners.shape[:2];times=np.asarray(times_s,dtype=float)
    present=np.asarray(frame_present)
    if times.shape!=(t,) or not np.isfinite(times).all() or (times<0).any() or (np.diff(times)<=0).any():
        raise ValueError('Actual times must be finite, nonnegative and strictly increasing')
    if present.shape!=(t,) or present.dtype!=bool or len(observations)!=t:
        raise ValueError('Frame-presence mask and observations must match times')
    query_valid=np.ones((p,t),dtype=bool) if candidate_geometry_valid is None else np.asarray(candidate_geometry_valid)
    if query_valid.shape!=(p,t) or query_valid.dtype!=bool:
        raise ValueError('Candidate geometry validity must be boolean [P,T]')
    values=np.full((p,t),np.nan);valid=np.zeros((p,t),dtype=bool)
    witness=np.full((p,t),'',dtype=object);invalid=np.zeros(t,dtype=int)
    for ti,objects in enumerate(observations):
        # Reject proxy sources even on a masked frame/object: no accidental
        # fallback can silently become a future training label.
        if any(o.source!='RECORDED_ANNOTATION' for o in objects):
            raise ValueError('Only RECORDED_ANNOTATION is accepted; no CV/future completion')
        ids=[o.actor_id for o in objects]
        if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
            raise ValueError('Each observed frame needs unique, nonempty actor IDs')
        if not present[ti]:
            if objects:raise ValueError('An absent frame cannot contain measured objects')
            continue
        usable=[]
        for obj in objects:
            if not obj.geometry_valid:
                invalid[ti]+=1;continue
            try:box=_ConvexBox(obj.vertices)
            except (ValueError,QhullError):
                invalid[ti]+=1;continue
            usable.append((obj.actor_id,box))
        for pi in range(p):
            if not query_valid[pi,ti]:continue
            try:query=_ConvexBox(corners[pi,ti])
            except (ValueError,QhullError):continue
            hit=next((actor for actor,box in usable if intersects(query,box)),None)
            if hit is not None:
                values[pi,ti]=0.;valid[pi,ti]=True;witness[pi,ti]=hit
            elif invalid[ti]==0:
                values[pi,ti]=1.;valid[pi,ti]=True
    failed=valid&(values==0)
    any_failed=failed.any(axis=1);all_passed=(valid&(values==1)).all(axis=1)
    trajectory_valid=any_failed|all_passed
    trajectory_value=np.where(any_failed,0.,np.where(all_passed,1.,np.nan))
    first=np.full(p,np.nan);first_valid=np.zeros(p,dtype=bool)
    for pi in range(p):
        if any_failed[pi]:
            ti=int(np.flatnonzero(failed[pi])[0])
            if valid[pi,:ti+1].all():first[pi]=times[ti];first_valid[pi]=True
        elif all_passed[pi]:first[pi]=np.inf;first_valid[pi]=True
    return GeometryTargets(trajectory_value,trajectory_valid,values,valid,first,first_valid,witness,invalid)
