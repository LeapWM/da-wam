"""Safety-first calibration losses for the PostTraj scorer."""

from typing import NamedTuple

import torch
import torch.nn.functional as F


class SafetyHardOutput(NamedTuple):
    weighted_final_loss: torch.Tensor
    rank_loss: torch.Tensor
    pair_count: torch.Tensor
    rank_accuracy: torch.Tensor
    unsafe_fraction: torch.Tensor
    unsafe_in_pred_topk: torch.Tensor
    selected_unsafe_rate: torch.Tensor
    safe_available_rate: torch.Tensor
    unsafe_overestimate_rate: torch.Tensor
    noc_failure_overestimate_rate: torch.Tensor
    dac_failure_overestimate_rate: torch.Tensor
    ttc_failure_overestimate_rate: torch.Tensor


def zero_safety_hard_output(reference: torch.Tensor) -> SafetyHardOutput:
    """Return a graph-preserving all-zero output."""
    zero = reference * 0.0
    metric_zero = zero.detach()
    return SafetyHardOutput(
        weighted_final_loss=zero,
        rank_loss=zero,
        pair_count=metric_zero,
        rank_accuracy=metric_zero,
        unsafe_fraction=metric_zero,
        unsafe_in_pred_topk=metric_zero,
        selected_unsafe_rate=metric_zero,
        safe_available_rate=metric_zero,
        unsafe_overestimate_rate=metric_zero,
        noc_failure_overestimate_rate=metric_zero,
        dac_failure_overestimate_rate=metric_zero,
        ttc_failure_overestimate_rate=metric_zero,
    )


def safety_hard_loss(
    pred_logits: torch.Tensor,
    target_scores: torch.Tensor,
    *,
    safety_threshold: float = 0.95,
    unsafe_final_weight: float = 4.0,
    topk: int = 8,
    margin: float = 0.05,
) -> SafetyHardOutput:
    """Reweight final BCE and penalize unsafe top-ranked items.

    Score order is ``[NOC, DAC, Progress, TTC, Comfort, Final]``.  The
    safety-rank term deliberately does not require the safe candidate to have
    a larger official final target.  If a scene contains any truly safe
    candidate, its highest-final safe candidate must outrank every truly
    unsafe item in the model's predicted top-k.
    """
    if pred_logits.ndim != 3 or pred_logits.shape[-1] < 6:
        raise ValueError("pred_logits must have shape [B,P,>=6]")
    if target_scores.ndim != 3 or target_scores.shape[-1] < 6:
        raise ValueError("target_scores must have shape [B,P,>=6]")
    if pred_logits.shape[:2] != target_scores.shape[:2]:
        raise ValueError("pred_logits and target_scores must align")
    if pred_logits.shape[1] == 0:
        raise ValueError("safety calibration requires at least one candidate")
    if not 0.0 <= safety_threshold <= 1.0:
        raise ValueError("safety_threshold must be within [0,1]")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if unsafe_final_weight < 1.0:
        raise ValueError("unsafe_final_weight must be >= 1")

    logits = pred_logits.float()
    scores = target_scores.detach().float()
    metric_indices = torch.tensor([0, 1, 3], device=scores.device)
    safety_targets = scores.index_select(-1, metric_indices)
    safe_metric = safety_targets >= float(safety_threshold)
    safe = safe_metric.all(dim=-1)
    final_logit = logits[..., 5]
    final_target = scores[..., 5]
    final_bce = F.binary_cross_entropy_with_logits(
        final_logit, final_target, reduction="none"
    )
    candidate_weight = torch.where(
        safe,
        torch.ones_like(final_target),
        final_target.new_full((), float(unsafe_final_weight)),
    )
    weighted_final_loss = (
        (final_bce * candidate_weight).sum()
        / candidate_weight.sum().clamp_min(1.0)
    )

    safe_available = safe.any(dim=1)
    pred_final = torch.sigmoid(final_logit)
    candidate_count = pred_final.shape[1]
    topk = min(int(topk), candidate_count)
    pred_top_index = torch.topk(
        pred_final.detach(), k=topk, dim=1, largest=True, sorted=False
    ).indices
    pred_top = torch.zeros_like(safe)
    pred_top.scatter_(1, pred_top_index, True)
    unsafe_pred_top = pred_top & ~safe

    safe_target_final = scores[..., 5].masked_fill(~safe, -torch.inf)
    best_safe_index = safe_target_final.argmax(dim=1, keepdim=True)
    best_safe_pred = pred_final.gather(1, best_safe_index)
    valid_pair = unsafe_pred_top & safe_available[:, None]
    pair_loss = torch.relu(
        float(margin) - (best_safe_pred - pred_final)
    )
    rank_loss = (
        (pair_loss * valid_pair.float()).sum()
        / valid_pair.sum().clamp_min(1)
    )
    rank_correct = best_safe_pred > pred_final
    rank_accuracy = (
        (rank_correct & valid_pair).sum().float()
        / valid_pair.sum().clamp_min(1)
    ).detach()

    selected = pred_final.detach().argmax(dim=1, keepdim=True)
    selected_safe = safe.gather(1, selected).squeeze(1)
    overestimate = pred_final.detach() > (final_target + 0.05)

    def conditional_rate(mask: torch.Tensor) -> torch.Tensor:
        return (
            (overestimate & mask).sum().float()
            / mask.sum().clamp_min(1)
        ).detach()

    return SafetyHardOutput(
        weighted_final_loss=weighted_final_loss,
        rank_loss=rank_loss,
        pair_count=valid_pair.sum().detach().float(),
        rank_accuracy=rank_accuracy,
        unsafe_fraction=(~safe).float().mean().detach(),
        unsafe_in_pred_topk=(
            unsafe_pred_top.sum(dim=1).float() / float(topk)
        ).mean().detach(),
        selected_unsafe_rate=(~selected_safe).float().mean().detach(),
        safe_available_rate=safe_available.float().mean().detach(),
        unsafe_overestimate_rate=conditional_rate(~safe),
        noc_failure_overestimate_rate=conditional_rate(~safe_metric[..., 0]),
        dac_failure_overestimate_rate=conditional_rate(~safe_metric[..., 1]),
        ttc_failure_overestimate_rate=conditional_rate(~safe_metric[..., 2]),
    )
