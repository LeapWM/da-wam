"""Recorded static props and annotated parked vehicles, sampled at 2 Hz.

Prepared hulls/AABBs come from the dense scene cache at exact 0.5 s knots.
No candidate interpolation or extrapolation of missing recorded objects.
"""
import numpy as np
from dataclasses import replace
from candidate_geometry_fast import lift
from prepared_geometry import score


def merge_collision(dynamic_value,dynamic_valid,static_value,static_valid):
    failed=(dynamic_valid&(dynamic_value==0))|(static_valid&(static_value==0))
    known=failed|(dynamic_valid&static_valid)
    return (~failed).astype(np.float32),known


def static_collision(proposals,prepared):
    p=np.asarray(proposals,dtype=np.float32)
    if p.ndim!=3 or len(p)<1 or p.shape[2]!=3 or p.shape[1] not in (6,8) or not np.isfinite(p).all():
        raise ValueError('Expected finite [P,6 or 8,3] proposals')
    scene=prepared['static_scene'];times=np.arange(1,p.shape[1]+1,dtype=float)*.5
    if len(scene.frames)!=p.shape[1]*5 or not np.allclose(scene.times_s,np.arange(1,len(scene.frames)+1)*.1,atol=1e-9,rtol=0):
        raise ValueError('Expected dense recorded 0.1 s scene for exact 0.5 s sampling')
    # Keep dense caches reusable; make a view of frames 5,10,...,30.
    sampled=replace(scene,frames=scene.frames[4::5],times_s=times,
                    fingerprint=scene.fingerprint+':sample_0.5s')
    geometry=lift(p,prepared['current_ego'],prepared['future_egos'][4::5],
                  world2ego=prepared['world2ego'],expert_trajectory=prepared['reference_dense'][4::5])
    result=score(geometry.corners,sampled,candidate_geometry_valid=geometry.available).labels
    return result.value,result.valid,result.first_overlap_s
