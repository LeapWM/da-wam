"""Exact-content, loss-call-local reuse between final scoring and object labels.

No persistent candidate caching. Results are copied on read because the final
scorer modifies collision clearance when merging the static gate.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import numpy as np

_scope=ContextVar('recorded_collision_scope',default=None)

@contextmanager
def collision_scope():
    state={'entries':{},'hits':0,'misses':0};token=_scope.set(state)
    try:yield state
    finally:_scope.reset(token)

def cached_safety(compute,point,actor_valid,actor_z,ego_z,dt,return_witness):
    state=_scope.get()
    if state is None:return compute(point,actor_valid,actor_z,ego_z,dt,return_witness)
    def exact(a):
        if a is None:return None
        a=np.asarray(a)
        return a.shape,a.dtype.str,a.tobytes()
    key=(float(dt),*(exact(a) for a in (point['fut_box_corners'],point['_ego_coords'],actor_valid,actor_z,ego_z)))
    if key not in state['entries']:
        state['entries'][key]=compute(point,actor_valid,actor_z,ego_z,dt,True);state['misses']+=1
    else:state['hits']+=1
    result=state['entries'][key]
    result=tuple(x.copy() for x in result)
    return result if return_witness else result[:3]
