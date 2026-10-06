"""Lazy per-process adapter from PostTraj training to the B2D v2 rule labels."""
import atexit
import gzip
import json
import os
from pathlib import Path
import pickle
import selectors
import subprocess
import threading

import numpy as np

from .b2d_rules_v2 import RuleConfig, VERSION, SUPPORTED_VERSIONS, score_scene
from .b2d_actor_geometry import GEOMETRY_VERSION


def align_actor_geometry(sample, geometry):
    """Match legacy rows to raw actor tracks without weakening geometry checks.

    The cache discarded actor IDs and rounded world2ego to float32. Actors
    with coincident first positions can consequently exchange distance-sort
    order. Require a one-to-one match of type, validity and ALL horizon boxes;
    never match independent frames or accept a changed/missing actor.
    """
    cached=np.asarray(sample['future_box_corners'],dtype=np.float32)
    rebuilt=np.asarray(geometry['reconstructed_box_corners'],dtype=np.float32)
    expected_valid=np.asarray(sample['future_box_valid'],dtype=bool)
    valid=np.asarray(geometry['reconstructed_box_valid'],dtype=bool)
    n=len(cached);ids=geometry['actor_ids']
    types=np.zeros(n,dtype=np.int8)
    if len(ids)>n:raise ValueError('Too many reconstructed actor IDs')
    for i,key in enumerate(ids):
        kind=key.partition(':')[0]
        if kind not in ('vehicle','walker'):raise ValueError('Unknown raw actor type: '+key)
        types[i]=1 if kind=='vehicle' else 2
    expected_types=np.asarray(sample['actor_types'])
    if rebuilt.shape!=cached.shape or valid.shape!=expected_valid.shape or expected_types.shape!=(n,):
        raise ValueError('Raw actor/cache shapes differ')
    if (np.allclose(rebuilt,cached,atol=2e-3,rtol=0)
            and np.array_equal(valid,expected_valid) and np.array_equal(types,expected_types)):
        return geometry
    matches=((np.abs(cached[:,None]-rebuilt[None])<=2e-3).all(axis=(2,3,4))
             & (expected_valid[:,None]==valid[None]).all(axis=2)
             & (expected_types[:,None]==types[None]))
    # Augmenting paths handle overlapping numerical tolerances without greedy
    # reuse of the same raw actor. Padded rows participate in the bijection too.
    owner=np.full(n,-1,dtype=int)
    def assign(row,seen):
        for raw in np.flatnonzero(matches[row]):
            if seen[raw]:continue
            seen[raw]=True
            if owner[raw]<0 or assign(owner[raw],seen):
                owner[raw]=row
                return True
        return False
    for row in np.argsort(matches.sum(axis=1),kind='stable'):
        if not assign(int(row),np.zeros(n,dtype=bool)):
            raise ValueError('No one-to-one raw actor/cache track match at row '+str(row))
    order=np.empty(n,dtype=int);order[owner]=np.arange(n)
    aligned=dict(geometry)
    for key in ('reconstructed_box_corners','reconstructed_box_valid','scoring_box_corners',
                'actor_height_overlap','actor_footprint_repaired','actor_z_bounds'):
        aligned[key]=np.asarray(geometry[key])[order].tolist()
    padded_ids=list(ids)+[None]*(n-len(ids))
    aligned['actor_ids']=[padded_ids[i] for i in order if padded_ids[i] is not None]
    aligned['actor_cache_reordered']=True
    aligned['actor_cache_permutation']=order.tolist()
    return aligned


class RoadClient:
    def __init__(self, python, carla_root, raw_root, metadata_root, timeout=180):
        self.args=[str(python),'-u',str(Path(__file__).with_name('b2d_road_worker.py')),
                   '--carla-root',str(carla_root),'--raw-root',str(raw_root),'--metadata-root',str(metadata_root)]
        self.timeout=timeout;self.process=None;self.owner=None;self.lock=threading.Lock()

    def close(self):
        if self.process is not None and self.owner==os.getpid():
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.process.stdin.close();self.process.stdout.close()
        self.process=None

    def query(self, sample, proposals, interval):
        with self.lock:
            if self.process is None or self.owner!=os.getpid():
                prefix=Path(self.args[0]).resolve().parent.parent
                env=os.environ.copy();env['PYTHONNOUSERSITE']='1'
                env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env.pop('LD_PRELOAD',None)
                env['LD_LIBRARY_PATH']=str(prefix/'lib/python3.8/site-packages/torch/lib')+':'+str(prefix/'lib')
                self.process=subprocess.Popen(self.args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
                self.owner=os.getpid();atexit.register(self.close)
            request=dict(token=sample['token'],town=sample['town_name'],world2ego=np.asarray(sample['world2ego']).tolist(),
                         expert_xy=np.asarray(sample['trajectory'])[:,:2].tolist(),proposals=np.asarray(proposals).tolist(),interval=interval,
                         max_agents=len(sample['future_box_corners']))
            self.process.stdin.write(json.dumps(request,allow_nan=False)+'\n');self.process.stdin.flush()
            with selectors.DefaultSelector() as sel:
                sel.register(self.process.stdout,selectors.EVENT_READ)
                if not sel.select(self.timeout):self.close();raise TimeoutError('B2D road worker timed out')
            line=self.process.stdout.readline()
            if not line:self.close();raise RuntimeError('B2D road worker exited')
            result=json.loads(line)
            if not result['ok']:raise RuntimeError(str(sample['token'])+': '+result['error'])
            if result['version']!=VERSION:raise RuntimeError('B2D rule version mismatch')
            if result.get('actor_geometry_version')!=GEOMETRY_VERSION:
                raise RuntimeError('B2D actor geometry worker version mismatch')
            return result


class B2DScoreProvider:
    def __init__(self, config):
        if config.b2d_scorer_version not in SUPPORTED_VERSIONS:
            raise ValueError('Explicit supported B2D label version required')
        self.config=config;self.cache={};self.road=None;self.pid=None

    def __getstate__(self):
        state=self.__dict__.copy();state.update(cache={},road=None,pid=None);return state

    def close(self):
        if self.road is not None:self.road.close()

    def score(self,tokens,proposals,training):
        if len(tokens)!=len(proposals):raise ValueError('Token and candidate batch sizes differ')
        path=self.config.b2d_train_scorer_cache_path if training else self.config.b2d_val_scorer_cache_path
        if not path:raise ValueError('Separate B2D train/validation scorer cache paths are required')
        if path not in self.cache:
            with gzip.open(path,'rb') as stream:data=pickle.load(stream)
            self.cache[path]=data.get('samples',data)
        if self.road is None or self.pid!=os.getpid():
            self.road=RoadClient(self.config.b2d_road_python,self.config.b2d_carla_root,self.config.b2d_raw_root,self.config.b2d_metadata_root)
            self.pid=os.getpid()
        results=[]
        for token,candidates in zip(tokens,proposals):
            sample=self.cache[path][str(token)]
            geometry=self.road.query(sample,candidates,self.config.trajectory_sampling.interval_length)
            try:
                geometry=align_actor_geometry(sample,geometry)
            except ValueError as error:
                raise ValueError('Raw actor identity/order/geometry does not match scorer cache: '
                                 +str(token)+': '+str(error)) from error
            rebuilt=np.asarray(geometry['reconstructed_box_corners'],dtype=np.float32)
            valid=np.asarray(geometry['reconstructed_box_valid'],dtype=bool)
            if (rebuilt.shape!=sample['future_box_corners'].shape
                    or not np.allclose(rebuilt,sample['future_box_corners'],atol=2e-3,rtol=0)
                    or not np.array_equal(valid,sample['future_box_valid'])):
                raise ValueError('Raw actor identity/order/geometry does not match scorer cache: '+str(token))
            height_overlap=np.asarray(geometry['actor_height_overlap'],dtype=bool)
            if height_overlap.shape!=valid.shape:raise ValueError('Actor height horizon mismatch')
            scoring_boxes=np.asarray(geometry['scoring_box_corners'],dtype=np.float32)
            if scoring_boxes.shape!=rebuilt.shape:raise ValueError('Scoring actor horizon mismatch')
            height_filtered_sample=dict(sample,future_box_valid=valid & height_overlap,
                                        future_box_corners=scoring_boxes)
            cfg=RuleConfig(score_version=self.config.b2d_scorer_version,
                interval=self.config.trajectory_sampling.interval_length,half_length=geometry['half_length'],
                half_width=geometry['half_width'],rear_axle_to_center=geometry['center_offset_x'],center_offset_y=geometry['center_offset_y'])
            try:
                result=score_scene(candidates,height_filtered_sample,geometry['road_mask'],geometry['initial_velocity'],cfg)
            except ValueError as error:
                raise ValueError(str(token)+': '+str(error)) from error
            result.update(actor_ids=geometry['actor_ids'],actor_height_overlap=height_overlap,
                          actor_cache_reordered=geometry.get('actor_cache_reordered',False),
                          actor_height_filtered=valid & ~height_overlap,
                          actor_footprint_repaired=geometry['actor_footprint_repaired'],
                          actor_geometry_version=GEOMETRY_VERSION)
            results.append(result)
        return results
