"""Pairwise ranking for the Formula scorer's normalized Progress score."""

from typing import NamedTuple

import torch


class ProgressRankingOutput(NamedTuple):
    loss: torch.Tensor
    pair_count: torch.Tensor
    accuracy: torch.Tensor


def progress_ranking_loss(
    pred_progress: torch.Tensor,
    target_progress: torch.Tensor,
    *,
    topk: int = 8,
    topk_weight: float = 1.0,
    margin_cap: float = 0.05,
    tie_epsilon: float = 1e-4,
) -> ProgressRankingOutput:
    """Rank EP scores and emphasize pairs whose better item is true top-k."""
    if pred_progress.shape != target_progress.shape:
        raise ValueError(
            "predicted and target Progress must align, got "
            f"{tuple(pred_progress.shape)} vs {tuple(target_progress.shape)}"
        )
    if pred_progress.ndim != 2:
        raise ValueError("Progress ranking expects [B,P] tensors")
    if pred_progress.shape[1] == 0:
        raise ValueError("Progress ranking requires at least one candidate")
    if topk <= 0 or topk_weight <= 0.0:
        raise ValueError("Progress top-k and weight must be positive")
    if margin_cap < 0.0 or tie_epsilon < 0.0:
        raise ValueError("Progress rank margin/epsilon must be non-negative")

    target_progress = target_progress.float()
    pred_progress = pred_progress.float()
    num_candidates = target_progress.shape[1]
    topk = min(int(topk), num_candidates)
    top_index = torch.topk(
        target_progress,
        k=topk,
        dim=1,
        largest=True,
        sorted=False,
    ).indices
    top_mask = torch.zeros_like(target_progress, dtype=torch.bool)
    top_mask.scatter_(1, top_index, True)

    target_diff = target_progress[:, :, None] - target_progress[:, None, :]
    pred_diff = pred_progress[:, :, None] - pred_progress[:, None, :]
    target_sign = torch.sign(target_diff)
    # Close EP pairs are retained and receive their actual target gap as the
    # margin. Large/easy gaps are capped so they do not dominate optimization.
    adaptive_margin = target_diff.abs().clamp(max=float(margin_cap))
    pair_loss = torch.relu(adaptive_margin - target_sign * pred_diff)

    higher_is_top = torch.where(
        target_diff > 0,
        top_mask[:, :, None],
        top_mask[:, None, :],
    )
    pair_weight = torch.where(
        higher_is_top,
        pair_loss.new_full((), float(topk_weight)),
        pair_loss.new_ones(()),
    )
    upper_triangle = torch.triu(
        torch.ones(
            num_candidates,
            num_candidates,
            device=target_progress.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    ).unsqueeze(0)
    valid_pair = upper_triangle & (target_diff.abs() > float(tie_epsilon))
    valid_weight = pair_weight * valid_pair
    loss = (
        (pair_loss * valid_weight).sum()
        / valid_weight.sum().clamp_min(1.0)
    )
    accuracy = (
        ((target_sign * pred_diff) > 0).float() * valid_pair
    ).sum() / valid_pair.sum().clamp_min(1)
    return ProgressRankingOutput(
        loss=loss,
        pair_count=valid_pair.sum().detach().float(),
        accuracy=accuracy.detach(),
    )


def safe_raw_progress_tie_ranking_loss(
    pred_raw_progress: torch.Tensor,
    target_raw_progress: torch.Tensor,
    target_scores: torch.Tensor,
    *,
    topk: int = 8,
    topk_weight: float = 1.0,
    safety_threshold: float = 0.95,
    ep_tie_epsilon: float = 1e-4,
    raw_tie_epsilon: float = 2e-4,
    margin_cap: float = 0.05,
) -> ProgressRankingOutput:
    """Rank raw distance only inside safe, official-EP-tied candidates.

    Raw progress is expected in normalized units. ``target_scores`` follows
    ``[NOC,DAC,EP,TTC,comfort,final]``. This auxiliary provides the ordering
    signal deliberately absent from the official five-metre full-score branch.
    """
    if pred_raw_progress.shape != target_raw_progress.shape:
        raise ValueError("predicted and target raw Progress must align")
    if pred_raw_progress.ndim != 2:
        raise ValueError("safe raw Progress ranking expects [B,P]")
    if (
        target_scores.shape[:2] != pred_raw_progress.shape
        or target_scores.shape[-1] < 4
    ):
        raise ValueError("target_scores must align and contain NOC/DAC/EP/TTC")
    if topk <= 0 or topk_weight <= 0.0:
        raise ValueError("Progress top-k and weight must be positive")
    if not 0.0 <= safety_threshold <= 1.0:
        raise ValueError("safety_threshold must be within [0,1]")
    if ep_tie_epsilon < 0.0 or raw_tie_epsilon < 0.0 or margin_cap < 0.0:
        raise ValueError("ranking epsilons/margin must be non-negative")

    pred = pred_raw_progress.float()
    target = target_raw_progress.float()
    scores = target_scores.float()
    safe = (
        (scores[..., 0] >= float(safety_threshold))
        & (scores[..., 1] >= float(safety_threshold))
        & (scores[..., 3] >= float(safety_threshold))
    )

    candidate_count = target.shape[1]
    topk = min(int(topk), candidate_count)
    safe_target = target.masked_fill(~safe, float("-inf"))
    top_index = torch.topk(
        safe_target, k=topk, dim=1, largest=True, sorted=False
    ).indices
    top_mask = torch.zeros_like(safe)
    top_mask.scatter_(1, top_index, True)

    target_diff = target[:, :, None] - target[:, None, :]
    pred_diff = pred[:, :, None] - pred[:, None, :]
    target_sign = torch.sign(target_diff)
    adaptive_margin = target_diff.abs().clamp(max=float(margin_cap))
    pair_loss = torch.relu(adaptive_margin - target_sign * pred_diff)
    higher_is_top = torch.where(
        target_diff > 0,
        top_mask[:, :, None],
        top_mask[:, None, :],
    )
    pair_weight = torch.where(
        higher_is_top,
        pair_loss.new_full((), float(topk_weight)),
        pair_loss.new_ones(()),
    )
    upper_triangle = torch.triu(
        torch.ones(
            candidate_count,
            candidate_count,
            device=target.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    ).unsqueeze(0)
    ep = scores[..., 2]
    ep_tied = (
        ep[:, :, None] - ep[:, None, :]
    ).abs() <= float(ep_tie_epsilon)
    valid_pair = (
        upper_triangle
        & safe[:, :, None]
        & safe[:, None, :]
        & ep_tied
        & (target_diff.abs() > float(raw_tie_epsilon))
    )
    valid_weight = pair_weight * valid_pair
    loss = (
        (pair_loss * valid_weight).sum()
        / valid_weight.sum().clamp_min(1.0)
    )
    accuracy = (
        ((target_sign * pred_diff) > 0).float() * valid_pair
    ).sum() / valid_pair.sum().clamp_min(1)
    return ProgressRankingOutput(
        loss=loss,
        pair_count=valid_pair.sum().detach().float(),
        accuracy=accuracy.detach(),
    )
