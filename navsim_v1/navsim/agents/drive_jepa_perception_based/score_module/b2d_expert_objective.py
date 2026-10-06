"""Dataset-expert imitation targets; no trajectory teacher or rule override."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def expert_quality(proposals, expert, scale_m=1.0, yaw_weight_m=0.5):
    if scale_m <= 0 or yaw_weight_m <= 0:
        raise ValueError('Positive imitation distance scales required')
    if proposals.ndim != 4 or proposals.shape[-1] != 3 or expert.shape != (proposals.shape[0], *proposals.shape[2:]):
        raise ValueError('Expected proposals [B,P,T,3] and dataset expert [B,T,3]')
    proposals=proposals.detach().float();expert=expert.detach().float()
    if not torch.isfinite(proposals).all() or not torch.isfinite(expert).all():
        raise ValueError('Nonfinite expert/proposal')
    delta=proposals-expert[:,None]
    xy=torch.linalg.vector_norm(delta[...,:2],dim=-1)
    yaw=torch.atan2(torch.sin(delta[...,2]),torch.cos(delta[...,2])).abs()
    distance=(xy+yaw_weight_m*yaw).mean(-1)
    # Expert has distance zero and quality one, irrespective of noisy risk labels.
    # Identical SE(2) trajectories may tie; no candidate-index special case.
    return 1.0/(1.0+distance/scale_m)


def matching_classification_loss(final_logits, quality):
    """Select best available imitation candidate, sharing targets across exact ties."""
    if final_logits.shape != quality.shape:
        raise ValueError('Logit and quality shapes must agree')
    best=quality.detach().amax(-1,keepdim=True)
    positive=(quality.detach()==best).to(final_logits.dtype)
    positive=positive/positive.sum(-1,keepdim=True)
    return -(positive*F.log_softmax(final_logits.float(),dim=-1)).sum(-1).mean()
