"""Candidate-related recorded vehicle/walker targets; no predicted actors.

Two channels: first collision witness, first aligned-TTC witness. After
selection, regress its observed future box corners in the current ego frame.
Absent actor observations are masked; witness selection uses the scorer's
discontinuity checks, rather than extrapolating missing actor states.
Static prop collisions remain in the final rule score, outside this actor head.
"""
import numpy as np
import torch
from torch import nn
from corrected_score import prepare_geometry,aligned_safety
from expert_gate import masked_final_bce

class CollisionObjectHead(nn.Module):
    def __init__(self,query_dim,future_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(query_dim+future_dim+3,query_dim),nn.LayerNorm(query_dim),
                               nn.GELU(),nn.Linear(query_dim,18))
    def forward(self,proposals,queries,future):
        x=torch.cat((queries,future,proposals.detach().to(queries.dtype)),-1)
        return self.net(x).reshape(*proposals.shape[:3],2,9)

def targets_for_sample(proposals,sample,prepared):
    point={'proposal':np.asarray(proposals,dtype=np.float32),'target_traj':sample['trajectory']}
    prepare_geometry(point,prepared)
    value,known,_,witness=aligned_safety(point,sample['future_box_valid'],
        prepared['actor_world_z'],prepared['ego_world_z'],return_witness=True)
    count,times=point['proposal'].shape[:2]
    boxes=np.zeros((count,times,2,8),np.float32);presence=np.zeros((count,times,2),np.float32)
    mask=np.zeros_like(presence,dtype=bool);boxmask=np.zeros_like(mask)
    recorded=prepared['scoring_box_corners']
    for pi in range(count):
        for channel in range(2):
            actor=witness[pi,channel]
            if actor<0:
                if known[pi,channel] and value[pi,channel]==1:mask[pi,:,channel]=True
                continue
            valid=np.asarray(sample['future_box_valid'][actor],bool)&np.isfinite(recorded[actor]).all(axis=(-1,-2))
            boxes[pi,valid,channel]=recorded[actor,valid].reshape(-1,8)
            presence[pi,valid,channel]=1;mask[pi,valid,channel]=True;boxmask[pi,valid,channel]=True
    return boxes,presence,mask,boxmask,witness[:count]

def auxiliary_loss(pred,proposals,targets):
    shape=pred.shape[:-1]
    boxes=np.zeros((*shape,8),np.float32);presence=np.zeros(shape,np.float32)
    mask=np.zeros(shape,bool);boxmask=np.zeros(shape,bool)
    array=proposals.detach().float().cpu().numpy()
    for i in range(len(array)):
        if targets['_scorer'][i] is None or targets['_safety'][i] is None:continue
        boxes[i],presence[i],mask[i],boxmask[i],_=targets_for_sample(array[i],targets['_scorer'][i],targets['_safety'][i])
    device=pred.device
    as_tensor=lambda x:torch.as_tensor(x,device=device)
    mask=as_tensor(mask);boxmask=as_tensor(boxmask)
    ce=masked_final_bce(pred[...,-1],as_tensor(presence),mask)
    error=(pred[...,:8].float()-as_tensor(boxes)).abs()
    total=torch.where(boxmask[...,None],error,0.).sum();count=boxmask.sum().to(total)*8
    global_count=count.detach().clone();global_sum=total.detach().clone()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(global_count);torch.distributed.all_reduce(global_sum)
        grad=total*torch.distributed.get_world_size()/global_count.clamp_min(1)
        regression=grad+(global_sum/global_count.clamp_min(1)-grad.detach())
    else:regression=total/count.clamp_min(1)
    return ce,regression,mask.sum().detach(),boxmask.sum().detach()
