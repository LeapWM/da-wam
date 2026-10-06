"""Fixed-horizon metrics, separate from partial-tail training diagnostics."""
import torch

def fixed_horizon_sums(selected,target,mask,weight):
    xy=(selected[...,:2]-target[...,:2]).float().norm(dim=-1)
    # iPad's released validation code includes yaw in its norm; retain that
    # reproduction separately, never label the mixed-unit value as XY meters.
    pose=(selected-target).float().norm(dim=-1)
    valid2=mask[:,:4].all(-1);valid3=mask.all(-1)
    w2=weight*valid2;w3=weight*valid3
    return torch.stack([(xy[:,:4].mean(-1)*w2).sum(),w2.sum(),
                        (xy.mean(-1)*w3).sum(),(xy[:,-1]*w3).sum(),w3.sum(),
                        (pose[:,:4].mean(-1)*w2).sum()]).double()
