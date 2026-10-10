"""Dense measured safety; no CV, actor interpolation, or expert-stop inference."""
import gzip,hashlib,json,os,pickle
import numpy as np
from pathlib import Path
from data import DenseDataset as BaseDataset,collate as base_collate
from horizon_geometry import prepare_sample
from metric_geometry import scoring_footprint
from bench2drive.training.scorer_cache import _select_actor_ids,_actors_by_id

class DenseDataset(BaseDataset):
    def __init__(self,root,index,split):
        super().__init__(root,index,split);self.split=split
        files=[Path(__file__),Path(__file__).with_name("metric_geometry.py")]+[Path('/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/geometry/src')/n for n in ('training_provider.py','recorded_geometry.py','prepared_geometry.py','candidate_geometry.py')]
        files.append(Path(__file__).with_name('horizon_geometry.py'))
        sig=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()[:16]
        self.safety_cache=self.root/'da_wam_3s_safety_cache'/sig
    def __getitem__(self,index):
        f,t=super().__getitem__(index);sample=t['_scorer'];clip=sample['clip'];start=sample['frame_id']
        dest=self.safety_cache/self.split/clip/f'{start:05d}.pkl.gz'
        try:
            with gzip.open(dest,'rb') as stream:prepared=pickle.load(stream)
        except FileNotFoundError:
            annotations={};hashes={}
            for i in range(start,start+31):
                p=self.root/'v1'/clip/'anno'/f'{i:05d}.json.gz'
                if not p.exists():continue
                raw=p.read_bytes();annotations[i]=json.loads(gzip.decompress(raw));hashes[i]=hashlib.sha256(raw).hexdigest()
            prepared=prepare_sample(sample,self.split,annotations,hashes)
            prepared['town']=sample['town_name']
            future=[annotations[start+i] for i in range(5,31,5)]
            actor_ids=_select_actor_ids(annotations[start],future,sample['world2ego'],len(sample['future_box_valid']))
            scoring_boxes=sample['future_box_corners'].copy();repaired=np.zeros(sample['future_box_valid'].shape,bool)
            actor_z=np.full((*sample['future_box_valid'].shape,2),np.nan)
            ego_z=[]
            for ti,a in enumerate(future):
                actors=_actors_by_id(a)
                for ai,actor_id in enumerate(actor_ids):
                    b=actors.get(actor_id)
                    if b is None:continue
                    corners=np.asarray(b.get('world_cord',[]))
                    if corners.shape==(8,3) and np.isfinite(corners).all():
                        actor_z[ai,ti]=[corners[:,2].min(),corners[:,2].max()]
                        scoring_boxes[ai,ti],repaired[ai,ti]=scoring_footprint(scoring_boxes[ai,ti],corners,sample['world2ego'])
                corners=np.asarray(next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')['world_cord'])
                ego_z.append([corners[:,2].min(),corners[:,2].max()])
            prepared['scoring_box_corners']=scoring_boxes;prepared['actor_footprint_repaired']=repaired
            prepared['actor_world_z']=actor_z;prepared['ego_world_z']=np.asarray(ego_z)
            # Keep every recorded light, not only lights affecting the expert.
            # Candidate lane association is checked against the static map.
            prepared['lights']=[None if i not in annotations else [{k:b.get(k) for k in ('id','state','rotation','trigger_volume_location','trigger_volume_rotation','trigger_volume_extent','road_id','lane_id')} for b in annotations[i]['bounding_boxes'] if b.get('class')=='traffic_light'] for i in range(start+1,start+31)]
            prepared['stops']=[None if i not in annotations else [b for b in annotations[i]['bounding_boxes'] if b.get('type_id')=='traffic.stop'] for i in range(start+1,start+31)]
            # Full observed prefix establishes whether a stop was already served.
            history=[]
            for i in range(sample.get('first_frame_id',0),start+1):
                path=self.root/'v1'/clip/'anno'/f'{i:05d}.json.gz'
                if not path.exists():continue
                with gzip.open(path,'rt') as stream:a=json.load(stream)
                ego=next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle')
                history.append({'frame':i,'position':ego['location'],'speed':a['speed'],'forward':np.linalg.inv(np.asarray(ego['world2ego']))[:3,0].tolist()})
            prepared['history']=history
            dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+f'.{os.getpid()}.tmp')
            with gzip.open(tmp,'wb',compresslevel=1) as stream:pickle.dump(prepared,stream,protocol=4)
            tmp.replace(dest)
        t['_safety']=prepared;return f,t

def collate(batch):
    prepared=[t['_safety'] for f,t in batch]
    clean=[(f,{k:v for k,v in t.items() if k!='_safety'}) for f,t in batch]
    f,t=base_collate(clean);t['_safety']=prepared;return f,t
