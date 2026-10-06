"""B2D-local training targets, versioned independently of NAVSIM/official DS.

No CARLA or torch import. Observed actors are log replay, not reactive rollouts.
Score order remains NOC, road, temporal_progress, TTC, comfort, final.
"""
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon, LineString, Point
from shapely.strtree import STRtree

VERSION = 'b2d_rules_v2.1'
PENALTY_VERSION = 'b2d_rules_v2.2'
TRACKING_VERSION = 'b2d_rules_v2.3'
SUPPORTED_VERSIONS = (VERSION, PENALTY_VERSION, TRACKING_VERSION)


@dataclass(frozen=True)
class RuleConfig:
    score_version: str = VERSION
    interval: float = 0.5
    half_length: float = 2.042
    half_width: float = 0.925
    rear_axle_to_center: float = 0.39
    center_offset_y: float = 0.0
    ttc_offsets: tuple = (0.5, 1.0)
    progress_epsilon: float = 0.01
    tracking_scale_m: float = 1.0

    def __post_init__(self):
        if self.score_version not in SUPPORTED_VERSIONS:
            raise ValueError('Unsupported B2D score version')
        if not np.isfinite(self.tracking_scale_m) or self.tracking_scale_m <= 0:
            raise ValueError('Tracking scale must be finite and positive')
        if min(self.interval, self.half_length, self.half_width, self.progress_epsilon) <= 0:
            raise ValueError('Lengths, interval and epsilon must be positive')
        for offset in self.ttc_offsets:
            if offset <= 0 or not np.isclose(offset/self.interval, round(offset/self.interval)):
                raise ValueError('TTC offsets must be positive multiples of interval')


def footprints(poses, config):
    poses = np.asarray(poses, dtype=np.float64)
    local = np.array([[config.half_length,config.half_width],[-config.half_length,config.half_width],
                      [-config.half_length,-config.half_width],[config.half_length,-config.half_width]])
    yaw=poses[...,2]; c=np.cos(yaw);s=np.sin(yaw)
    result=np.empty((*poses.shape[:-1],4,2))
    result[...,0]=poses[...,0,None]+config.rear_axle_to_center*c[...,None]-config.center_offset_y*s[...,None]+c[...,None]*local[:,0]-s[...,None]*local[:,1]
    result[...,1]=poses[...,1,None]+config.rear_axle_to_center*s[...,None]+config.center_offset_y*c[...,None]+s[...,None]*local[:,0]+c[...,None]*local[:,1]
    return result


def _collisions(ego_corners, boxes, valid):
    result=np.zeros(ego_corners.shape[:2],dtype=bool)
    actor_hits=np.zeros((ego_corners.shape[0],boxes.shape[0]),dtype=bool)
    for t in range(ego_corners.shape[1]):
        ids=np.flatnonzero(valid[:,t])
        if not len(ids):continue
        geometries=[Polygon(boxes[a,t]) for a in ids]
        tree=STRtree(geometries)
        if int(shapely.__version__.split('.')[0])>=2:
            hit=tree.query([Polygon(x) for x in ego_corners[:,t]],predicate='intersects')
            result[hit[0],t]=True
            actor_hits[hit[0],ids[hit[1]]]=True
        else:
            # CARLA's Python 3.8 environment uses Shapely 1.x: query returns
            # geometry objects filtered by bounding box, so test intersections.
            actor_index={id(g):a for g,a in zip(geometries,ids)}
            for p,corners in enumerate(ego_corners[:,t]):
                footprint=Polygon(corners)
                for other in tree.query(footprint):
                    if footprint.intersects(other):
                        result[p,t]=True
                        actor_hits[p,actor_index[id(other)]]=True
    return result,actor_hits


def temporal_progress(proposals, expert, epsilon):
    """Match progress at every execution time; stopped expert remains a valid target."""
    xy=np.vstack([np.zeros((1,2)),expert[:,:2]])
    distance=np.linalg.norm(np.diff(xy,axis=0),axis=-1)
    if distance.sum() <= 1e-8:
        # Degenerate centerline: moving away from a waiting expert is not rewarded.
        candidate_distance=np.linalg.norm(proposals[:,:,:2],axis=-1)
        return (epsilon/(candidate_distance+epsilon)).mean(axis=-1)
    # Repeated standstill samples make GEOS divide by zero on zero-length segments.
    # Removing consecutive duplicates preserves the reference polyline exactly.
    line=LineString(xy[np.r_[True,np.any(np.diff(xy,axis=0)!=0,axis=1)]])
    ref=np.array([line.project(Point(p)) for p in expert[:,:2]])
    candidate=np.array([[line.project(Point(p)) for p in path[:,:2]] for path in proposals])
    ratios=(np.minimum(candidate,ref)+epsilon)/(np.maximum(candidate,ref)+epsilon)
    # Equal fixed time weights remove the endpoint-only delayed-action tie.
    return ratios.mean(axis=-1)


def tracking_progress(proposals, expert, epsilon=.01, scale_m=1.0):
    """Time-aligned progress times distance-to-finite-reference penalty.

    Distance includes lateral departure AND longitudinal endpoint overshoot.
    The fixed 1 m default is an explicit imitation-proximity scale, not B2D DS.
    Geometry is vectorized over candidates and reference segments.
    """
    xy=np.vstack([np.zeros((1,2)),np.asarray(expert)[:,:2]])
    xy=xy[np.r_[True,np.any(np.diff(xy,axis=0)!=0,axis=1)]]
    points=np.asarray(proposals)[...,:2]
    if len(xy)==1:
        return (1/(1+np.linalg.norm(points,axis=-1)/scale_m)).mean(axis=-1)
    vectors=np.diff(xy,axis=0);length=np.linalg.norm(vectors,axis=-1)
    def project(query):
        delta=query[...,None,:]-xy[:-1]
        fraction=np.clip((delta*vectors).sum(-1)/(length**2),0,1)
        residual=delta-fraction[...,None]*vectors
        squared=(residual**2).sum(-1)
        nearest=squared.argmin(-1)[...,None]
        distance=np.sqrt(np.take_along_axis(squared,nearest,-1)[...,0])
        arc=np.r_[0,np.cumsum(length)[:-1]]+fraction*length
        return np.take_along_axis(arc,nearest,-1)[...,0],distance
    candidate,distance=project(points)
    ref,_=project(np.asarray(expert)[:,:2])
    ratio=(np.minimum(candidate,ref)+epsilon)/(np.maximum(candidate,ref)+epsilon)
    return (ratio/(1+distance/scale_m)).mean(axis=-1)


def select_rule_candidate(scores, model_scores, prefer_safe=True):
    """Offline rule selection; true geometry labels are required, not logits.

    Filter to NOC=road=1 when possible. With no such candidate, use the soft
    local utility as a deterministic fallback, without claiming it is safe.
    """
    scores=np.asarray(scores);model_scores=np.asarray(model_scores)
    if scores.ndim!=2 or scores.shape[1]!=6 or len(scores)==0 or model_scores.shape!=(len(scores),):
        raise ValueError('Expected nonempty [P,6] scores and [P] model scores')
    if not np.isfinite(scores).all() or not np.isfinite(model_scores).all():
        raise ValueError('Nonfinite selection inputs')
    safe=(scores[:,0]==1)&(scores[:,1]==1)
    ids=np.flatnonzero(safe) if prefer_safe and safe.any() else np.arange(len(scores))
    return int(ids[np.lexsort((ids,-model_scores[ids],-scores[ids,-1]))[0]])


def score_scene(proposals, sample, road_mask, initial_velocity, config=RuleConfig()):
    proposals=np.asarray(proposals,dtype=np.float64)
    expert=np.asarray(sample['trajectory'],dtype=np.float64)
    boxes=np.asarray(sample['future_box_corners'],dtype=np.float64)
    valid=np.asarray(sample['future_box_valid'],dtype=bool)
    road_mask=np.asarray(road_mask,dtype=bool)
    initial_velocity=np.asarray(initial_velocity,dtype=np.float64)
    if proposals.ndim!=3 or proposals.shape[-1]!=3 or proposals.shape[1]<3:
        raise ValueError('Expected [P,T>=3,3] candidates')
    p,t,_=proposals.shape
    if expert.shape!=(t,3) or road_mask.shape!=(p,t) or boxes.shape!=(len(boxes),t,4,2) or valid.shape!=boxes.shape[:2]:
        raise ValueError('Candidate, reference, actor and road horizons must agree')
    if initial_velocity.shape!=(2,) or not all(np.isfinite(a).all() for a in [proposals,expert,initial_velocity,boxes[valid]]):
        raise ValueError('Nonfinite input or missing measured initial velocity')
    for value in boxes[valid]:
        if not Polygon(value).is_valid or Polygon(value).area<=0:
            raise ValueError('A valid actor box must be a nondegenerate polygon')
    corners=footprints(proposals,config)
    collision,actor_hits=_collisions(corners,boxes,valid)
    # No expert collision subtraction: every candidate retains its own actor events.
    xy=np.concatenate([np.zeros((p,1,2)),proposals[:,:,:2]],axis=1)
    velocity=np.diff(xy,axis=1)/config.interval
    ttc=np.zeros((p,t),dtype=bool);ttc_valid=np.zeros(t,dtype=bool)
    for offset in config.ttc_offsets:
        shift=int(round(offset/config.interval))
        if shift>=t:continue
        # Ego and actors are both queried at time t+offset; no frozen other actor.
        extrapolated=proposals[:,:-shift].copy()
        extrapolated[:,:,:2]+=velocity[:,:-shift]*offset
        hits,_=_collisions(footprints(extrapolated,config),boxes[:,shift:],valid[:,shift:])
        ttc[:,:-shift]|=hits;ttc_valid[:-shift]=True
    acceleration=np.diff(np.concatenate([np.broadcast_to(initial_velocity,(p,1,2)),velocity],axis=1),axis=1)/config.interval
    jerk=np.diff(acceleration,axis=1)/config.interval
    yaw=np.unwrap(np.concatenate([np.zeros((p,1)),proposals[:,:,2]],axis=1),axis=1)
    yaw_rate=np.diff(yaw,axis=1)/config.interval
    yaw_accel=np.diff(yaw_rate,axis=1)/config.interval
    comfort=((np.linalg.norm(acceleration,axis=-1)<4.89).all(axis=1)
             &(np.linalg.norm(jerk,axis=-1)<8.37).all(axis=1)
             &(np.abs(yaw_rate)<.95).all(axis=1)&(np.abs(yaw_accel)<1.93).all(axis=1))
    noc=~collision.any(axis=1);road=road_mask.all(axis=1)
    progress=temporal_progress(proposals,expert,config.progress_epsilon)
    if config.score_version == TRACKING_VERSION:
        progress=tracking_progress(proposals,expert,config.progress_epsilon,config.tracking_scale_m)
    # Training utility, not official DS. Comfort/TTC are separate diagnostics/auxiliaries.
    # A moving reference gives a stationary plan near-zero utility instead of a 7/12 floor.
    final=noc*road*progress
    collision_penalty=noc.astype(np.float64)
    if config.score_version in (PENALTY_VERSION, TRACKING_VERSION):
        if t*config.interval>5:
            raise ValueError('v2.2 actor event aggregation requires a horizon <= 5 s')
        # B2D official DS severity, applied to local geometric risk events.
        # Keep NOC binary. This proxy does not reconstruct collision sensors,
        # responsibility, spatial event deduplication, full-route DS, or SR.
        actor_types=np.asarray(sample.get('actor_types',[]))
        if actor_types.shape!=(len(boxes),) or not np.isin(actor_types[valid.any(axis=1)],[1,2]).all():
            raise ValueError('v2.2 requires aligned vehicle=1/walker=2 actor types')
        weights=np.where(actor_types==2,.5,.6)
        # Once per actor inside the 4 s horizon, rather than once per sample.
        # The official same-actor event suppression window is 5 s.
        collision_penalty=np.prod(np.where(actor_hits,weights[None,:],1.),axis=1)
        final=collision_penalty*road*progress
    scores=np.stack([noc,road,progress,~ttc.any(axis=1),comfort,final],axis=-1).astype(np.float32)
    return dict(version=config.score_version,scores=scores,collision_penalty=collision_penalty,
                collision_by_time=collision,collision_actor_hits=actor_hits,
                ttc_by_time=ttc,ttc_valid_by_time=ttc_valid,road_by_time=road_mask,
                acceleration=acceleration)
