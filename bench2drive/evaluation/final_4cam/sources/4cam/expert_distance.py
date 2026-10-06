"""Relative expert matching targets, not calibrated safety probabilities."""
import torch

def distance_targets(proposals,expert):
    error=(proposals[...,:2]-expert[:,None,:,:2]).float().norm(dim=-1)
    distance=.5*(error.mean(-1)+error[...,-1])
    maximum=distance.amax(-1,keepdim=True)
    score=(1-distance/maximum.clamp_min(1e-6)).clamp(0,1)
    score=torch.where(maximum<=1e-6,torch.ones_like(score),score)
    return score.detach(),distance.detach()
