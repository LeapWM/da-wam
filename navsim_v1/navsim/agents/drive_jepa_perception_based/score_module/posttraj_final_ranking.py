"""Safety-aware hard ranking for the PostTraj learned final-score head."""

from typing import NamedTuple

import torch


class FinalTopKRankingOutput(NamedTuple):
    loss: torch.Tensor
    pair_count: torch.Tensor
    accuracy: torch.Tensor
    topk_overlap: torch.Tensor
    true_best_in_pred_topk: torch.Tensor
    unsafe_in_pred_topk: torch.Tensor
    safety_pair_count: torch.Tensor
    safety_pair_accuracy: torch.Tensor


def final_topk_hard_ranking_loss(
    pred_final: torch.Tensor,
    target_scores: torch.Tensor,
    *,
    topk: int = 8,
    score_gap: float = 0.01,
    margin_cap: float = 0.05,
    false_topk_weight: float = 2.0,
    safety_pair_weight: float = 3.0,
    safety_threshold: float = 0.95,
) -> FinalTopKRankingOutput:
    """Rank the union of true and predicted top-k candidates.

    ``pred_final`` is the final-head probability ``[B,P]``.  Target scores
    follow ``[NOC,DAC,EP,TTC,comfort,final]``.  Candidate identities are
    preserved: the function never pairs independently sorted values.
    """
    if pred_final.ndim != 2:
        raise ValueError("pred_final must have shape [B,P]")
    if target_scores.ndim != 3 or target_scores.shape[-1] < 6:
        raise ValueError("target_scores must have shape [B,P,>=6]")
    if target_scores.shape[:2] != pred_final.shape:
        raise ValueError("pred_final and target_scores must align")
    if pred_final.shape[1] == 0:
        raise ValueError("ranking requires at least one candidate")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if score_gap < 0.0 or margin_cap < 0.0:
        raise ValueError("score_gap and margin_cap must be non-negative")
    if false_topk_weight <= 0.0 or safety_pair_weight <= 0.0:
        raise ValueError("pair weights must be positive")
    if not 0.0 <= safety_threshold <= 1.0:
        raise ValueError("safety_threshold must be within [0,1]")

    pred = pred_final.float()
    scores = target_scores.detach().float()
    target = scores[..., 5]
    candidate_count = pred.shape[1]
    topk = min(int(topk), candidate_count)

    true_top_index = torch.topk(
        target, k=topk, dim=1, largest=True, sorted=False
    ).indices
    pred_top_index = torch.topk(
        pred.detach(), k=topk, dim=1, largest=True, sorted=False
    ).indices
    true_top = torch.zeros_like(target, dtype=torch.bool)
    pred_top = torch.zeros_like(target, dtype=torch.bool)
    true_top.scatter_(1, true_top_index, True)
    pred_top.scatter_(1, pred_top_index, True)
    union = true_top | pred_top

    target_diff = target[:, :, None] - target[:, None, :]
    pred_diff = pred[:, :, None] - pred[:, None, :]
    valid_pair = (
        union[:, :, None]
        & union[:, None, :]
        & (target_diff > float(score_gap))
    )

    adaptive_margin = target_diff.clamp(min=0.0, max=float(margin_cap))
    pair_loss = torch.relu(adaptive_margin - pred_diff)
    pair_weight = torch.ones_like(pair_loss)

    # A true top-k item should outrank a predicted top-k false positive.
    false_topk_pair = (
        true_top[:, :, None]
        & pred_top[:, None, :]
        & ~true_top[:, None, :]
    )
    pair_weight = torch.where(
        false_topk_pair,
        pair_weight.new_full((), float(false_topk_weight)),
        pair_weight,
    )

    safe = (
        (scores[..., 0] >= float(safety_threshold))
        & (scores[..., 1] >= float(safety_threshold))
        & (scores[..., 3] >= float(safety_threshold))
    )
    safety_pair = safe[:, :, None] & ~safe[:, None, :] & valid_pair
    pair_weight = torch.where(
        safety_pair,
        torch.maximum(
            pair_weight,
            pair_weight.new_full((), float(safety_pair_weight)),
        ),
        pair_weight,
    )

    valid_weight = pair_weight * valid_pair
    loss = (
        (pair_loss * valid_weight).sum()
        / valid_weight.sum().clamp_min(1.0)
    )
    correct = pred_diff > 0.0
    pair_count = valid_pair.sum().detach().float()
    accuracy = (
        (correct & valid_pair).sum().float()
        / valid_pair.sum().clamp_min(1)
    ).detach()

    topk_overlap = (
        (true_top & pred_top).sum(dim=1).float() / float(topk)
    ).mean().detach()
    true_best = target.argmax(dim=1, keepdim=True)
    true_best_in_pred_topk = (
        pred_top.gather(1, true_best).float().mean().detach()
    )
    unsafe_in_pred_topk = (
        (pred_top & ~safe).sum(dim=1).float() / float(topk)
    ).mean().detach()
    safety_pair_count = safety_pair.sum().detach().float()
    safety_pair_accuracy = (
        (correct & safety_pair).sum().float()
        / safety_pair.sum().clamp_min(1)
    ).detach()

    return FinalTopKRankingOutput(
        loss=loss,
        pair_count=pair_count,
        accuracy=accuracy,
        topk_overlap=topk_overlap,
        true_best_in_pred_topk=true_best_in_pred_topk,
        unsafe_in_pred_topk=unsafe_in_pred_topk,
        safety_pair_count=safety_pair_count,
        safety_pair_accuracy=safety_pair_accuracy,
    )
