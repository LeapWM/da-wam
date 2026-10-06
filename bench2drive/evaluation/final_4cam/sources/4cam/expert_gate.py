"""Masked final-score BCE; expert decisions live in corrected_score.py."""
import torch
import torch.nn.functional as F

def masked_final_bce(logits,targets,keep):
    mask=keep[:,None].expand_as(logits) if keep.ndim==1 else keep
    if not torch.isfinite(targets[mask]).all():raise ValueError('Nonfinite retained score target')
    targets=torch.where(mask,targets,torch.zeros_like(targets))
    values=F.binary_cross_entropy_with_logits(logits.float(),targets.float(),reduction='none')
    total=(values*mask).sum();count=mask.sum().to(total)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        global_count=count.detach().clone();global_sum=total.detach().clone()
        torch.distributed.all_reduce(global_count);torch.distributed.all_reduce(global_sum)
        grad_term=total*torch.distributed.get_world_size()/global_count.clamp_min(1)
        return grad_term+(global_sum/global_count.clamp_min(1)-grad_term.detach())
    return total/count.clamp_min(1)
