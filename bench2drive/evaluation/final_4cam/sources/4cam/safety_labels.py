import atexit,json,os,selectors,subprocess
from pathlib import Path
import numpy as np
from candidate_geometry import lift,MIRROR,transform
from horizon_geometry import dense_queries
from prepared_geometry import score

CHANNELS=('recorded_collision_clear','sampled_road_clear','recorded_actor_clear_1s','comfort')
AUX_CHANNELS=('annotated_static_clear_4s','nonjunction_lane_direction','recorded_red_light_compliance')

def known_and(a,av,b,bv):
    fail=(av&(a==0))|(bv&(b==0));clear=av&bv&(a==1)&(b==1)
    return np.where(fail,0.,np.where(clear,1.,np.nan)),fail|clear

class RoadWorker:
    def __init__(self,script=None):
        self.process=None;self.pid=None;self.script=Path(script) if script else Path(__file__).with_name('road_worker.py')
    def close(self):
        if self.process is not None and self.pid==os.getpid():
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process=None
    def request(self,q):
        if self.process is None or self.pid!=os.getpid():
            prefix=os.environ.get(
                'B2D_ROAD_ENV',
                '/mnt/c2-worldmodel/2639639/Bench2Drive/envs/road_py38_carla0915',
            );env=dict(os.environ)
            for k in ('PYTHONPATH','PYTHONHOME','LD_PRELOAD'):env.pop(k,None)
            env['LD_LIBRARY_PATH']=prefix+'/lib/python3.8/site-packages/torch/lib:'+prefix+'/lib';env['PYTHONNOUSERSITE']='1'
            self.process=subprocess.Popen([prefix+'/bin/python','-u',str(self.script)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
            self.pid=os.getpid();atexit.register(self.close)
        self.process.stdin.write(json.dumps(q,allow_nan=False)+'\n');self.process.stdin.flush()
        with selectors.DefaultSelector() as sel:
            sel.register(self.process.stdout,selectors.EVENT_READ)
            if not sel.select(180):self.close();raise TimeoutError('Safety map worker timed out')
        line=self.process.stdout.readline()
        if not line:self.close();raise RuntimeError('Safety map worker exited')
        r=json.loads(line)
        if not r.get('ok'):raise RuntimeError(r)
        return r

class Labels:
    def __init__(self,traffic_only=False):
        self.road=RoadWorker();self.traffic_only=traffic_only
    def score(self,proposals,samples):
        metrics=[];masks=[];aux=[];aux_masks=[];first=[];hashes=[]
        for p,sample in zip(np.asarray(proposals,dtype=np.float32),samples):
            # Single selected-candidate validation uses the same original32
            # geometry code by tiling; only the requested positions are kept.
            count=len(p)
            if count not in (1,32):raise ValueError('Expected 1 or32 candidates')
            full=np.repeat(p,32,axis=0) if count==1 else p
            dense=dense_queries(full)
            g=lift(dense,sample['current_ego'],sample['future_egos'],world2ego=sample['world2ego'],expert_trajectory=sample['reference_dense'])
            if not self.traffic_only:
                static=score(g.corners,sample['static_scene'],candidate_geometry_valid=g.available).labels
                near=score(g.corners[:,:10],sample['near_scene'],candidate_geometry_valid=g.available[:,:10]).labels
                noc,noc_valid=known_and(static.value,static.valid,near.value,near.valid)
            # Condition only candidate height/tilt on recorded ego, explicitly
            # preserving the inherited label scope. Actors are never inferred.
            w2e=sample['world2ego'];inverse=MIRROR@np.linalg.inv(w2e)
            z=np.array([np.nan if e is None else transform(e['location'],w2e@MIRROR)[2] for e in sample['future_egos']])
            local=np.concatenate((dense[:count,:,:2],np.broadcast_to(z,(count,len(z)))[...,None]),axis=-1)
            world=transform(local,inverse)[...,:3]
            forward=np.stack((np.cos(dense[:count,:,2]),np.sin(dense[:count,:,2]),np.zeros_like(dense[:count,:,2])),-1)@inverse[:3,:3].T
            corners=np.nan_to_num(g.corners[:count]);world=np.nan_to_num(world)
            # Ego half length from exact current box in the recorded ego frame.
            ec=transform(np.asarray(sample['current_ego']['world_cord']),w2e@MIRROR)[:,:3]
            half=(ec[:,0].max()-ec[:,0].min())/2
            result=self.road.request({'town':sample['town'],'traffic_only':self.traffic_only,'corners':[] if self.traffic_only else corners.tolist(),'world_positions':world.tolist(),'world_forward':forward.tolist(),'available':g.available[:count].tolist(),'lights':sample['lights'],'stops':sample['stops'],'history':sample['history'],'half_length':float(half)})
            rv=np.asarray(result['value']);rm=np.asarray(result['valid'],bool)
            if self.traffic_only:
                # Uncomputed channels remain unknown, never safe labels.
                metrics.append(np.zeros((count,3),np.float32));masks.append(np.zeros((count,3),bool))
                aux.append(np.stack((np.zeros(count),rv[:,1],rv[:,2]),-1))
                aux_masks.append(np.stack((np.zeros(count,bool),rm[:,1],rm[:,2]),-1))
                first.append(np.full(count,np.nan));hashes.append(result['map_sha256'])
                continue
            metrics.append(np.stack((noc[:count],rv[:,0],near.value[:count]),-1))
            masks.append(np.stack((noc_valid[:count],rm[:,0],near.valid[:count]),-1))
            aux.append(np.stack((static.value[:count],rv[:,1],rv[:,2]),-1));aux_masks.append(np.stack((static.valid[:count],rm[:,1],rm[:,2]),-1))
            first.append(near.first_overlap_s[:count]);hashes.append(result['map_sha256'])
        return {'value':np.nan_to_num(np.asarray(metrics),nan=0.).astype(np.float32),'valid':np.asarray(masks),
                'aux_value':np.nan_to_num(np.asarray(aux),nan=0.).astype(np.float32),'aux_valid':np.asarray(aux_masks),
                'first_recorded_overlap_s':np.asarray(first),'map_sha256':hashes}
