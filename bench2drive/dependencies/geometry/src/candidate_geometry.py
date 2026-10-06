"""Lift existing V1 local XY/yaw proposals using measured reference geometry.

This retains the dataset's tilted ego frame and relative-yaw convention. The
reference's unpredicted height/tilt is an explicit condition of the metric, NOT
ground-truth terrain under an arbitrary different candidate. No actor future
or new candidate is generated here. Use only in the training-label producer.
"""
from dataclasses import dataclass

import numpy as np


MIRROR=np.diag([1.,-1.,1.,1.])


@dataclass(frozen=True)
class LiftedGeometry:
    corners: np.ndarray             # [original32,T,8,3], CARLA world frame
    available: np.ndarray           # [original32,T], finite recorded inputs
    reference_trajectory: np.ndarray # [T,3], recomputed dataset convention
    condition: str = 'recorded_ego_height_and_relative_tilt'
    candidate_terrain_certified: bool = False


def transform(points, matrix):
    points=np.asarray(points,dtype=np.float64)
    return np.concatenate((points,np.ones(points.shape[:-1]+(1,))),axis=-1)@matrix.T


def lift(proposals, current_ego, future_egos, *, world2ego, expert_trajectory):
    """Preserve original32; validate cache/raw identity before creating labels.

    Future entries correspond exactly to supplied proposal times, with None
    representing missing annotations. Callers bind timestamp/hash provenance.
    'available' confirms finite matching inputs, not counterfactual road support.
    """
    p=np.asarray(proposals,dtype=np.float64);w2e=np.asarray(world2ego,dtype=np.float64)
    expert=np.asarray(expert_trajectory,dtype=np.float64)
    if p.ndim!=3 or p.shape[0]!=32 or p.shape[-1]!=3 or p.shape[1]==0:
        raise ValueError('Expected original32 [32,T,3] proposals')
    t=p.shape[1]
    if len(future_egos)!=t or expert.shape!=(t,3):raise ValueError('Reference horizon mismatch')
    expected=MIRROR@np.asarray(current_ego['world2ego'],dtype=np.float64)@MIRROR
    if w2e.shape!=(4,4) or not np.isfinite(w2e).all() or not np.allclose(w2e,expected,atol=2e-3,rtol=0):
        raise ValueError('Scorer cache/current annotation coordinate mismatch')
    if not np.allclose(w2e[3],[0,0,0,1],atol=1e-7,rtol=0):raise ValueError('Expected affine world2ego')
    world_to_local=w2e@MIRROR;local_to_world=MIRROR@np.linalg.inv(w2e)
    corners=np.full((32,t,8,3),np.nan);available=np.zeros((32,t),dtype=bool)
    reference=np.full((t,3),np.nan)
    for ti,future in enumerate(future_egos):
        if future is None:continue
        if str(future['id'])!=str(current_ego['id']):raise ValueError('Ego identity changed within reference')
        vertices=np.asarray(future.get('world_cord',[]),dtype=np.float64)
        location=np.asarray(future.get('location',[]),dtype=np.float64)
        raw_w2e=np.asarray(future.get('world2ego',[]),dtype=np.float64)
        if vertices.shape!=(8,3) or location.shape!=(3,) or raw_w2e.shape!=(4,4):continue
        if not all(np.isfinite(x).all() for x in (vertices,location,raw_w2e)):continue
        local_position=transform(location,world_to_local)[:3]
        relative=w2e@np.linalg.inv(MIRROR@raw_w2e@MIRROR)
        yaw=np.arctan2(relative[1,0],relative[0,0])
        reference[ti]=[local_position[0],local_position[1],yaw]
        if not np.isfinite(expert[ti]).all() or not np.allclose(reference[ti,:2],expert[ti,:2],atol=.002,rtol=0):
            raise ValueError('Scorer expert XY does not match this reference frame')
        yaw_error=np.arctan2(np.sin(yaw-expert[ti,2]),np.cos(yaw-expert[ti,2]))
        if abs(yaw_error)>1e-5:raise ValueError('Scorer expert yaw convention/reference mismatch')
        offset=transform(vertices,world_to_local)[:,:3]-local_position
        delta=p[:,ti,2]-yaw;c=np.cos(delta);s=np.sin(delta)
        local=np.empty((32,8,3))
        local[:,:,0]=c[:,None]*offset[None,:,0]-s[:,None]*offset[None,:,1]+p[:,ti,0,None]
        local[:,:,1]=s[:,None]*offset[None,:,0]+c[:,None]*offset[None,:,1]+p[:,ti,1,None]
        local[:,:,2]=offset[None,:,2]+local_position[2]
        finite=np.isfinite(p[:,ti]).all(axis=-1)
        corners[finite,ti]=transform(local[finite],local_to_world)[:,:,:3]
        available[finite,ti]=True
    return LiftedGeometry(corners,available,reference)
