"""Official split, dense eligible frames, real future only, lazy numeric cache."""
import functools,gzip,hashlib,json,os,pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from bench2drive.training.scorer_cache import ScorerCacheConfig,_build_sample
from bench2drive.training.da_wam_cache import preprocess_front_jpeg,front_lidar_to_processed_image,build_ego_status

@functools.lru_cache(maxsize=64)
def annotation(path):
    with gzip.open(path,'rt') as f:return json.load(f)

def index_clip(folder):
    folder=Path(folder);ids=sorted(int(p.name.split('.')[0]) for p in (folder/'anno').glob('*.json.gz'))
    available=set(ids)
    eligible=[i for i in ids if all(i+k in available for k in range(5,31,5))]
    return {'folder':folder.name,'raw':len(ids),'first':min(ids),'frames':eligible,'excluded_missing_future':len(ids)-len(eligible)}

class DenseDataset(Dataset):
    def __init__(self,root,index,split):
        self.root=Path(root);self.entries=[]
        for c in index[split]:self.entries.extend((c['folder'],i,c['first']) for i in c['frames'])
        sources=[Path(__file__),Path(__import__('bench2drive.training.scorer_cache',fromlist=['x']).__file__),Path(__import__('bench2drive.training.da_wam_cache',fromlist=['x']).__file__)]
        signature=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:16]
        self.cache=self.root/'da_wam_3s_numeric_cache'/signature
    def __len__(self):return len(self.entries)
    def __getitem__(self,idx):
        weight=1.
        if isinstance(idx,tuple):idx,weight=idx
        clip,frame,first=self.entries[idx];folder=self.root/'v1'/clip
        cache=self.cache/clip/f'{frame:05d}.pkl.gz'
        try:
            with gzip.open(cache,'rb') as f:sample,projection,status=pickle.load(f)
        except FileNotFoundError:
            annotations={i:annotation(str(folder/'anno'/f'{i:05d}.json.gz')) for i in [frame]+list(range(frame+5,frame+31,5))}
            sample=_build_sample(clip,frame,annotations,ScorerCacheConfig(horizon=6,sample_stride=1))
            projection=front_lidar_to_processed_image(annotations[frame]);status=build_ego_status(annotations[frame],6,'scalar_speed')
            cache.parent.mkdir(parents=True,exist_ok=True);tmp=cache.with_name(cache.name+f'.{os.getpid()}.tmp')
            with gzip.open(tmp,'wb',compresslevel=1) as f:pickle.dump((sample,projection,status),f,protocol=4)
            tmp.replace(cache)
        def image(i):return preprocess_front_jpeg((folder/'camera/rgb_front'/f'{i:05d}.jpg').read_bytes())
        features={'camera_feature_1':image(frame),'camera_feature_2':image(max(first,frame-5)),
                  'ego_status':status,'lidar2img':projection,'img_shape':torch.tensor([[256.,512.,3.]]*4)}
        targets={'trajectory':torch.as_tensor(sample['trajectory'],dtype=torch.float32),'token':sample['token'],
                 'future_camera_features':image(frame+5).unsqueeze(0),'future_camera_offset_order':torch.tensor([1]),
                 'sample_weight':torch.tensor(weight),'_scorer':sample}
        return features,targets

def collate(batch):
    features,targets=zip(*batch)
    samples=[t['_scorer'] for t in targets]
    out=default_collate([{k:v for k,v in t.items() if k!='_scorer'} for t in targets]);out['_scorer']=samples
    return default_collate(features),out

class ValidationSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self,dataset,rank,world):
        super().__init__(dataset,num_replicas=world,rank=rank,shuffle=False)
        self.size=len(dataset);self.rank=rank;self.world=world
    def __len__(self):return (self.size+self.world-1)//self.world
    def __iter__(self):
        for p in range(self.rank,len(self)*self.world,self.world):yield p%self.size,float(p<self.size)
