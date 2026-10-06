"""Safe-candidate final-score ranking for PostTraj Stage 2."""

from typing import NamedTuple

import torch


class SafeFinalRankingOutput(NamedTuple):
    loss: torch.Tensor
    pair_count: torch.Tensor
    accuracy: torch.Tensor
    safe_candidate_fraction: torch.Tensor
    safe_available_rate: torch.Tensor


def zero_safe_final_ranking_output(
    reference: torch.Tensor,
) -> SafeFinalRankingOutput:
    """Return a graph-preserving all-zero output."""
    zero = reference * 0.0
    metric_zero = zero.detach()
    return SafeFinalRankingOutput(
        loss=zero,
        pair_count=metric_zero,
        accuracy=metric_zero,
        safe_candidate_fraction=metric_zero,
        safe_available_rate=metric_zero,
    )


def safe_final_ranking_loss(
    pred_logits: torch.Tensor,
    target_scores: torch.Tensor,
    *,
    safety_threshold: float = 0.95,
    topk: int = 8,
    score_gap: float = 0.01,
    margin_cap: float = 0.05,
) -> SafeFinalRankingOutput:
    """Rank the true best-final safe candidate over predicted high safe ones.

    Score order is ``[NC, DAC, Progress, TTC, Comfort, Final]``.  Unsafe
    candidates never enter this term; they remain covered by safety-hard BCE
    and safe-vs-unsafe ranking.  The pair margin follows the true final-score
    gap and is capped to keep the auxiliary rank term small.
    """
    if pred_logits.ndim != 3 or pred_logits.shape[-1] < 6:
        raise ValueError("pred_logits must have shape [B,P,>=6]")
    if target_scores.ndim != 3 or target_scores.shape[-1] < 6:
        raise ValueError("target_scores must have shape [B,P,>=6]")
    if pred_logits.shape[:2] != target_scores.shape[:2]:
        raise ValueError("pred_logits and target_scores must align")
    if pred_logits.shape[1] == 0:
        raise ValueError("safe ranking requires at least one candidate")
    if not 0.0 <= safety_threshold <= 1.0:
        raise ValueError("safety_threshold must be within [0,1]")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if score_gap < 0.0 or margin_cap < 0.0:
        raise ValueError("score_gap and margin_cap must be non-negative")

    logits = pred_logits.float()
    scores = target_scores.detach().float()
    metric_indices = torch.tensor([0, 1, 3], device=scores.device)
    safe = (
        scores.index_select(-1, metric_indices)
        >= float(safety_threshold)
    ).all(dim=-1)
    safe_available = safe.any(dim=1)
    pred_final = torch.sigmoid(logits[..., 5])
    target_final = scores[..., 5]

    safe_target_final = target_final.masked_fill(~safe, -torch.inf)
    best_safe_index = safe_target_final.argmax(dim=1, keepdim=True)
    best_safe_target = target_final.gather(1, best_safe_index)
    best_safe_pred = pred_final.gather(1, best_safe_index)

    # Restrict hard mining to model-preferred safe candidates.  This directly
    # trains the decision boundary used after unsafe candidates are rejected.
    safe_pred = pred_final.detach().masked_fill(~safe, -torch.inf)
    candidate_count = pred_final.shape[1]
    topk = min(int(topk), candidate_count)
    pred_top_index = torch.topk(
        safe_pred, k=topk, dim=1, largest=True, sorted=False
    ).indices
    pred_top = torch.zeros_like(safe)
    pred_top.scatter_(1, pred_top_index, True)

    target_gap = best_safe_target - target_final
    valid_pair = (
        pred_top
        & safe
        & safe_available[:, None]
        & (target_gap >= float(score_gap))
    )
    margin = target_gap.clamp(min=0.0, max=float(margin_cap))
    pair_loss = torch.relu(margin - (best_safe_pred - pred_final))
    pair_count = valid_pair.sum()
    loss = (
        (pair_loss * valid_pair.float()).sum()
        / pair_count.clamp_min(1)
    )
    accuracy = (
        ((best_safe_pred > pred_final) & valid_pair).sum().float()
        / pair_count.clamp_min(1)
    ).detach()

    return SafeFinalRankingOutput(
        loss=loss,
        pair_count=pair_count.detach().float(),
        accuracy=accuracy,
        safe_candidate_fraction=safe.float().mean().detach(),
        safe_available_rate=safe_available.float().mean().detach(),
    )
