"""Train-only original32 targets from versioned measured-geometry caches.

This prototype emits three explicitly new channels; it is not a replacement
for the current six-channel scorer. No actor forecasting, CV fallback, new
candidate or trajectory teacher is implemented. Runtime inference never calls
this provider. Positive risk labels retain their recorded-box scope.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from candidate_geometry import MIRROR, transform, lift
from prepared_geometry import prepare, score
from recorded_geometry import ObservedBox


VERSION='recorded_safety_provider_v1'
CHANNELS=('annotated_static_clear_4s','recorded_actor_clear_1s','expert_similarity')
SOURCES=('STATIC_GEOMETRY','RECORDED_FUTURE','EXPERT_TRAJECTORY')
DENSE_TIMES=np.arange(1,41,dtype=float)*.1
KNOT_TIMES=np.arange(9,dtype=float)*.5


def _ego(annotation):
    records=[b for b in annotation['bounding_boxes'] if b['class']=='ego_vehicle']
    if len(records)!=1:raise ValueError('Expected one recorded ego per frame')
    b=records[0]
    return {k:b[k] for k in ('id','location','world2ego','world_cord')}


def prepare_sample(sample,split,annotations,annotation_hashes):
    """annotations/hash map uses integer raw frame IDs, never padded futures."""
    if split not in ('train','val'):raise ValueError('Explicit train/val split required')
    token=str(sample['token']);clip,f=token.rsplit('/',1);f=int(f)
    if sample['clip']!=clip or int(sample['frame_id'])!=f:raise ValueError('Scorer token/frame mismatch')
    if f not in annotations:raise ValueError('Current annotation is required')
    current=_ego(annotations[f]);w2e=np.asarray(sample['world2ego'],dtype=float)
    expected=MIRROR@np.asarray(current['world2ego'],dtype=float)@MIRROR
    if w2e.shape!=(4,4) or not np.isfinite(w2e).all() or not np.allclose(w2e,expected,atol=.002,rtol=0):
        raise ValueError('Current raw/scorer coordinate mismatch')
    expert=np.asarray(sample['trajectory'],dtype=np.float32)
    if expert.ndim!=2 or expert.shape[1]!=3 or len(expert) not in (6,8) or not np.isfinite(expert).all():raise ValueError('Malformed expert knots')
    dense_steps=len(expert)*5;dense_times=np.arange(1,dense_steps+1,dtype=float)*.1
    future=[];reference=np.full((dense_steps,3),np.nan);corners=np.full((dense_steps,8,3),np.nan)
    present=[];static=[];actors=[];hashes=[]
    if f not in annotation_hashes:raise ValueError('Current frame provenance required')
    for i in range(1,dense_steps+1):
        ann=annotations.get(f+i);present.append(ann is not None)
        if ann is None:
            if f+i in annotation_hashes:raise ValueError('Missing frame has a declared content hash')
            future.append(None);hashes.append(None);static.append([]);actors.append([]);continue
        if f+i not in annotation_hashes:raise ValueError('Observed frame lacks source hash')
        hashes.append(annotation_hashes[f+i]);ego=_ego(ann);future.append(ego)
        if str(ego['id'])!=str(current['id']):raise ValueError('Ego ID changed within clip window')
        position=transform(ego['location'],w2e@MIRROR)[:3]
        relative=w2e@np.linalg.inv(MIRROR@np.asarray(ego['world2ego'],dtype=float)@MIRROR)
        reference[i-1]=[position[0],position[1],np.arctan2(relative[1,0],relative[0,0])]
        raw=np.asarray(ego['world_cord'],dtype=float)
        if raw.shape!=(8,3) or not np.isfinite(raw).all():raise ValueError('Malformed recorded ego corners')
        corners[i-1]=raw;static_frame=[];actor_frame=[]
        for b in ann['bounding_boxes']:
            prop=str(b.get('type_id','')).startswith('static.prop.')
            parked=b['class']=='vehicle' and b.get('state')=='static'
            if prop or parked:
                static_frame.append(ObservedBox(b['class']+':'+str(b['id']),np.asarray(b.get('world_cord',[]))))
            # Includes parked/temporarily stopped vehicles as well: absence of
            # a 'static' tag cannot remove them from the near-actor component.
            if b['class'] in ('vehicle','walker'):
                actor_frame.append(ObservedBox(b['class']+':'+str(b['id']),np.asarray(b.get('world_cord',[]))))
        static.append(static_frame);actors.append(actor_frame)
    present=np.asarray(present,dtype=bool)
    for j,i in enumerate(range(4,dense_steps,5)):
        if not present[i]:continue
        xy=np.abs(reference[i,:2]-expert[j,:2]).max()
        yaw=abs(np.arctan2(np.sin(reference[i,2]-expert[j,2]),np.cos(reference[i,2]-expert[j,2])))
        if xy>.002 or yaw>1e-5:raise ValueError('Expert target does not match raw future')
    return dict(version=VERSION,token=token,split=split,frame_ids=list(range(f+1,f+dense_steps+1)),
        annotation_hashes=hashes,current_annotation_hash=annotation_hashes[f],
        current_ego=current,future_egos=future,world2ego=w2e,expert_knots=expert,reference_dense=reference,
        static_scene=prepare(corners,static,times_s=dense_times,frame_present=present),
        near_scene=prepare(corners[:10],actors[:10],times_s=DENSE_TIMES[:10],frame_present=present[:10]),
        static_object_time_observations=sum(map(len,static)),near_actor_time_observations=sum(map(len,actors[:10])),
        scope=f'Static props + annotated parked vehicles over {len(expert)*.5:g} s; all recorded vehicle/walker boxes over 1 s',
        candidate_height_condition='recorded_ego_height_and_relative_tilt',world_safety_certified=False)


def dense_queries(proposals):
    p=np.asarray(proposals,dtype=np.float32)
    if p.ndim!=3 or p.shape[0]!=32 or p.shape[2]!=3 or p.shape[1] not in (6,8) or not np.isfinite(p).all():raise ValueError('Expected finite [32,6 or 8,3] proposals')
    dense_times=np.arange(1,p.shape[1]*5+1,dtype=float)*.1
    knot_times=np.arange(p.shape[1]+1,dtype=float)*.5
    knots=np.concatenate((np.zeros((32,1,3)),p.astype(float)),axis=1)
    knots[:,:,2]=np.unwrap(knots[:,:,2],axis=1)
    result=np.empty((32,p.shape[1]*5,3))
    for pi in range(32):
        for c in range(3):result[pi,:,c]=np.interp(dense_times,knot_times,knots[pi,:,c])
    # Candidate interpolation is explicit; no actor future is interpolated.
    np.testing.assert_allclose(result[:,4::5,:2],p[:,:,:2],atol=1e-10,rtol=0)
    return result


@dataclass(frozen=True)
class BatchTargets:
    value: np.ndarray                 # [B,32,3]
    valid: np.ndarray
    source_names: tuple
    channels: tuple
    first_overlap_s: np.ndarray       # [B,32,2]; inf is right-censored clear
    first_overlap_valid: np.ndarray
    time_targets: tuple


class TrainingLabelProvider:
    def __init__(self,manifest_path,*,allow_diagnostic_subset=False,max_in_memory=16):
        self.path=Path(manifest_path).resolve();self.root=self.path.parent
        self.manifest=json.loads(self.path.read_text());self.cache=OrderedDict();self.limit=int(max_in_memory)
        if self.manifest.get('version')!=VERSION or tuple(self.manifest.get('channels',[]))!=CHANNELS:
            raise ValueError('Explicit compatible safety cache version/channels required')
        if tuple(self.manifest.get('source_names',[]))!=SOURCES or self.manifest.get('cv_or_actor_interpolation') is not False:
            raise ValueError('Measured-source contract required; no CV/actor interpolation')
        if not self.manifest.get('full_training_coverage',False) and not allow_diagnostic_subset:
            raise ValueError('Diagnostic subset cannot be used as a full training cache')
        if set(self.manifest.get('samples',{}))!={'train','val'} or self.limit<1:raise ValueError('Separate train/val indices and positive cache limit required')
        if set(self.manifest['samples']['train'])&set(self.manifest['samples']['val']):raise ValueError('Train/val token overlap')
        clips={s:{t.rsplit('/',1)[0] for t in self.manifest['samples'][s]} for s in ('train','val')}
        if clips['train']&clips['val']:raise ValueError('Train/val clip overlap')
        contract=dict(candidate_count=32,model_knots=8,model_knot_interval_s=.5,
            actual_annotation_interval_s=.1,static_horizon_s=4.,near_actor_horizon_s=1.)
        if any(self.manifest.get(k)!=v for k,v in contract.items()):raise ValueError('Manifest trajectory/time contract mismatch')

    def _load(self,token,training):
        split='train' if training else 'val';index=self.manifest['samples'][split]
        if token not in index:raise KeyError('Token unavailable in requested '+split+' split: '+token)
        entry=index[token];key=(split,token,entry['sha256'])
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        path=(self.root/entry['path']).resolve()
        if self.root not in path.parents:raise ValueError('Cache entry escapes its root')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Safety cache content hash mismatch')
        sample=pickle.loads(raw)
        if sample.get('version')!=VERSION or sample.get('token')!=token or sample.get('split')!=split:
            raise ValueError('Safety cache token/split/version mismatch')
        start=int(entry['frame_start'])
        if sample.get('frame_ids')!=list(range(start+1,start+41)) or len(sample.get('annotation_hashes',[]))!=40:
            raise ValueError('Safety cache raw frame sequence mismatch')
        if not np.array_equal(sample['static_scene'].times_s,DENSE_TIMES) or not np.array_equal(sample['near_scene'].times_s,DENSE_TIMES[:10]):
            raise ValueError('Safety cache physical horizon mismatch')
        if (sample['static_scene'].fingerprint!=entry['static_scene_fingerprint']
                or sample['near_scene'].fingerprint!=entry['near_scene_fingerprint']):
            raise ValueError('Safety cache geometry fingerprint mismatch')
        observed=[h is not None for h in sample['annotation_hashes']]
        if observed!=[f.present for f in sample['static_scene'].frames] or observed[:10]!=[f.present for f in sample['near_scene'].frames]:
            raise ValueError('Safety cache observation/source presence mismatch')
        self.cache[key]=sample
        if len(self.cache)>self.limit:self.cache.popitem(last=False)
        return sample

    def score(self,tokens,proposals,*,training):
        if type(training) is not bool:raise ValueError('Explicit boolean training split required')
        p=np.asarray(proposals,dtype=np.float32)
        if p.shape!=(len(tokens),32,8,3) or not len(tokens) or not np.isfinite(p).all():raise ValueError('Expected finite [B,32,8,3] original candidates')
        values=[];valid=[];first=[];first_valid=[];time_targets=[]
        for token,candidates in zip(tokens,p):
            sample=self._load(str(token),training);dense=dense_queries(candidates)
            geometry=lift(dense,sample['current_ego'],sample['future_egos'],world2ego=sample['world2ego'],expert_trajectory=sample['reference_dense'])
            static=score(geometry.corners,sample['static_scene'],candidate_geometry_valid=geometry.available).labels
            near=score(geometry.corners[:,:10],sample['near_scene'],candidate_geometry_valid=geometry.available[:,:10]).labels
            delta=candidates-sample['expert_knots'][None]
            xy=np.linalg.norm(delta[...,:2],axis=-1)
            yaw=np.abs(np.arctan2(np.sin(delta[...,2]),np.cos(delta[...,2])))
            quality=1./(1.+(xy+.5*yaw).mean(axis=-1))
            values.append(np.stack((static.value,near.value,quality),axis=-1))
            valid.append(np.stack((static.valid,near.valid,np.ones(32,dtype=bool)),axis=-1))
            first.append(np.stack((static.first_overlap_s,near.first_overlap_s),axis=-1))
            first_valid.append(np.stack((static.first_overlap_valid,near.first_overlap_valid),axis=-1))
            time_targets.append((static,near))
        return BatchTargets(np.asarray(values,dtype=np.float32),np.asarray(valid,dtype=bool),SOURCES,CHANNELS,
            np.asarray(first),np.asarray(first_valid),tuple(time_targets))
