"""Geometry helpers ported from the audited b2d_rules_v2 branch (2026-09-07)."""

import numpy as np

def scoring_footprint(legacy_xy, raw_corners, world2ego):
    """Keep normal cached footprints; recover collapsed faces from all 8 corners.

    A fallen actor's local XY face can project to a line although its full 3D
    box has a finite footprint. The enclosing rectangle preserves the [4,2]
    cache interface and conservatively includes the entire projected box.
    No invented minimum size and no removal of the actor is used.
    """
    xy = np.asarray(legacy_xy, dtype=np.float64)
    shifted = xy - xy[0]
    twice_area = abs(np.sum(shifted[:, 0] * np.roll(shifted[:, 1], -1)
                            - shifted[:, 1] * np.roll(shifted[:, 0], -1)))
    if twice_area > 2e-8:
        return xy, False
    from shapely.geometry import MultiPoint
    points = transform_points(raw_corners, world2ego)[:, :2]
    hull = MultiPoint(points).convex_hull
    if hull.geom_type != 'Polygon' or not hull.is_valid or hull.area <= 1e-8:
        raise ValueError('Full raw 3D actor box also has a degenerate XY footprint')
    repaired = np.asarray(hull.minimum_rotated_rectangle.exterior.coords)[:4]
    return repaired, True

def transform_points(points, world2ego):
    points = np.asarray(points).copy()
    points[..., 1] *= -1
    return (np.c_[points, np.ones(len(points))] @ np.asarray(world2ego).T)[:, :3]

def footprints(poses, config):
    poses = np.asarray(poses, dtype=np.float64)
    local = np.array([[config.half_length,config.half_width],[-config.half_length,config.half_width],
                      [-config.half_length,-config.half_width],[config.half_length,-config.half_width]])
    yaw=poses[...,2]; c=np.cos(yaw);s=np.sin(yaw)
    result=np.empty((*poses.shape[:-1],4,2))
    result[...,0]=poses[...,0,None]+config.rear_axle_to_center*c[...,None]-config.center_offset_y*s[...,None]+c[...,None]*local[:,0]-s[...,None]*local[:,1]
    result[...,1]=poses[...,1,None]+config.rear_axle_to_center*s[...,None]+config.center_offset_y*c[...,None]+s[...,None]*local[:,0]+c[...,None]*local[:,1]
    return result

def measured_velocity(history,world2ego):
    if len(history)<2:return None
    a,b=history[-2:]
    if b['frame']-a['frame']!=1:return None
    velocity=(np.asarray(b['position'])-a['position'])/.1
    if not np.isfinite(velocity).all() or np.linalg.norm(velocity)>100:return None
    velocity[1]*=-1
    return (np.asarray(world2ego)[:3,:3]@velocity)[:2]

def initial_state_comfort(proposals,initial_velocity,dt=.5):
    if initial_velocity is None:return np.ones(len(proposals),bool),np.zeros(len(proposals),bool)
    xy=np.concatenate([np.zeros((len(proposals),1,2)),proposals[...,:2]],axis=1)
    velocity=np.diff(xy,axis=1)/dt
    acceleration=np.diff(np.concatenate([np.broadcast_to(initial_velocity,(len(proposals),1,2)),velocity],axis=1),axis=1)/dt
    jerk=np.diff(acceleration,axis=1)/dt
    yaw=np.unwrap(np.concatenate([np.zeros((len(proposals),1)),proposals[...,2]],axis=1),axis=1)
    rate=np.diff(yaw,axis=1)/dt
    value=(np.linalg.norm(acceleration,axis=-1)<4.89).all(-1)&(np.linalg.norm(jerk,axis=-1)<8.37).all(-1)&(np.abs(rate)<.95).all(-1)&(np.abs(np.diff(rate,axis=1)/dt)<1.93).all(-1)
    return value,np.isfinite(proposals).all(axis=(1,2))


def expert_relative_comfort(proposals,expert,dt=.5):
    """iPad design: acceleration and turn rate bounded by this sample's expert.

    Use our ego-forward yaw=0 convention. Unlike upstream's batch-wide max
    and strict '<', equality (including a stopped expert) passes. Wrapped yaw
    differences avoid a spurious 2*pi turn. No actor prediction is involved.
    """
    p=np.asarray(proposals,dtype=np.float64);e=np.asarray(expert,dtype=np.float64)
    if p.ndim!=3 or p.shape[1:]!=e.shape or e.shape[-1]!=3 or len(e)<2:
        raise ValueError('Expected matching [P,T,3] and [T,3] trajectories')
    valid=np.isfinite(p).all(axis=(1,2))&np.isfinite(e).all()
    trajectories=np.concatenate([p,e[None]])
    velocity=np.diff(trajectories[...,:2],axis=1,prepend=np.zeros((len(trajectories),1,2)))/dt
    acceleration=np.linalg.norm(np.diff(velocity,axis=1),axis=-1)/dt
    heading_delta=np.diff(trajectories[...,2],axis=1,prepend=np.zeros((len(trajectories),1)))
    turn=np.abs(np.arctan2(np.sin(heading_delta),np.cos(heading_delta)))/dt
    amax=acceleration[-1].max();rmax=turn[-1].max()
    value=(acceleration[:-1]<=amax+1e-6).all(-1)&(turn[:-1]<=rmax+1e-6).all(-1)
    return value&valid,valid
