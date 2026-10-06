"""All observed frames; only recorded future points receive supervision."""
import gzip,json,os,pickle
from pathlib import Path
import numpy as np
import torch
from safety_data import DenseDataset as CompleteDataset,collate
from data import annotation
from bench2drive.training.scorer_cache import _world2ego,_future_ego_pose
from bench2drive.training.drive_jepa_cache import preprocess_front_jpeg,front_lidar_to_processed_image,build_ego_status
from route_targets import RouteTargetCache

TAIL_CACHE_SIGNATURE='scalar-speed-v1'


def expand_index(index):
    """Accepted data has continuous raw frame IDs; do not alter the old index."""
    expanded=json.loads(json.dumps(index))
    expanded['contract'].update(future_steps=6,future_dt=.5,future_horizon_s=3.)
    expanded['contract']['missing_future']='retain frame; mask missing labels; rule score requires all 6 points'
    for split in ('train','val'):
        for c in expanded[split]:
            old=c['frames'];all_ids=list(range(c['first'],c['first']+c['raw']))
            if old!=all_ids[:-40]:raise ValueError('Expected verified continuous 10Hz clip with 40 excluded tail frames')
            c['complete_frames']=all_ids[:-30];c['frames']=all_ids;c['excluded_missing_future']=0
    return expanded


class DenseDataset(CompleteDataset):
    def __init__(self,root,index,split,load_route_targets=True):
        super().__init__(root,index,split)
        self.split=split
        self.route_targets=RouteTargetCache() if load_route_targets else None
        self.ends={c['folder']:c['first']+c['raw']-1 for c in index[split]}
        self.complete={c['folder']:set(c.get('complete_frames',c['frames'])) for c in index[split]}
        # Route-target loading does not change trajectory/mask cache content.
        # Change this only when the tail numeric generation contract changes.
        self.tail_cache=self.root/'ema_jqtf_tail_numeric_cache'/TAIL_CACHE_SIGNATURE

    def _route_features(self,features,clip,frame):
        if self.route_targets is None:return features
        route=self.route_targets.get(self.split,clip,frame)
        features['target_point']=torch.as_tensor(route['near_xy'],dtype=torch.float32)
        features['target_point_valid']=torch.tensor(bool(route['near_valid']))
        return features

    def __getitem__(self,index):
        idx,weight=index if isinstance(index,tuple) else (index,1.)
        clip,frame,first=self.entries[idx]
        if frame in self.complete[clip]:
            f,t=super().__getitem__(index)
            self._route_features(f,clip,frame)
            t.update(trajectory_valid=torch.ones(6,dtype=torch.bool),
                     future_camera_valid=torch.ones(1,dtype=torch.bool),rule_score_valid=torch.tensor(True))
            return f,t
        folder=self.root/'v1'/clip;last=self.ends[clip]
        path=self.tail_cache/clip/f'{frame:05d}.pkl.gz'
        try:
            with gzip.open(path,'rb') as f:trajectory,mask,projection,status=pickle.load(f)
        except FileNotFoundError:
            current=annotation(str(folder/'anno'/f'{frame:05d}.json.gz'))
            w2e=_world2ego(current);trajectory=np.zeros((6,3),np.float32);mask=np.zeros(6,bool)
            for ti,offset in enumerate(range(5,31,5)):
                if frame+offset<=last:
                    # Missing a supposedly observed annotation is corruption, not padding.
                    future=annotation(str(folder/'anno'/f'{frame+offset:05d}.json.gz'))
                    trajectory[ti]=_future_ego_pose(w2e,current,future);mask[ti]=True
            projection=front_lidar_to_processed_image(current);status=build_ego_status(current,6,'scalar_speed')
            path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
            with gzip.open(tmp,'wb',compresslevel=1) as f:pickle.dump((trajectory,mask,projection,status),f,protocol=4)
            tmp.replace(path)
        def image(i):return preprocess_front_jpeg((folder/'camera/rgb_front'/f'{i:05d}.jpg').read_bytes())
        current=image(frame);has_future=bool(mask[0])
        f={'camera_feature_1':current,'camera_feature_2':image(max(first,frame-5)),
           'ego_status':status,'lidar2img':projection,'img_shape':torch.tensor([[256.,512.,3.]]*4)}
        self._route_features(f,clip,frame)
        t={'trajectory':torch.as_tensor(trajectory),'trajectory_valid':torch.as_tensor(mask),
           'future_camera_features':(image(frame+5) if has_future else torch.zeros_like(current))[None],
           'future_camera_valid':torch.tensor([has_future]),'future_camera_offset_order':torch.tensor([1]),
           'rule_score_valid':torch.tensor(False),'sample_weight':torch.tensor(weight),
           'token':f'{clip}/{frame:05d}','_scorer':None,'_safety':None}
        return f,t
