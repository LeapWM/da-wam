"""NAVSIM v2 EPDMS composition and scorer post-training utilities.

This module intentionally has no nuPlan dependency.  It can therefore be used by
training, diagnostics, and unit tests without importing the full NAVSIM runtime.
The metric order and weights mirror NAVSIM v2's official ``PDMScorerConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


EPDMS_METRICS: Tuple[str, ...] = (
    "NC",
    "DAC",
    "DDC",
    "TLC",
    "EP",
    "TTC",
    "LK",
    "HC",
    "EC",
)
MULTIPLICATIVE_METRIC_COUNT = 4
WEIGHTED_METRIC_WEIGHTS: Tuple[float, ...] = (5.0, 5.0, 2.0, 2.0, 2.0)
EXTENDED_COMFORT_INDEX = EPDMS_METRICS.index("EC")


def drivor_v2_selection_score(
    metric_values: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Rank trajectories with DrivoR's log-domain rule extended to NAVSIM v2.

    DrivoR ranks with the logarithm of the official multiplicative/weighted
    score.  For v2, DDC and TLC join NC and DAC as multiplicative metrics, and
    LK/HC join EP/TTC as weighted metrics.  EC is intentionally absent because
    it is computed across adjacent outputs by the official SceneAggregator and
    is not available while selecting an isolated frame.

    The omitted weighted-metric normalization is constant across proposals, so
    this function has the same argmax as official stage-1 EPDMS composition.
    """

    if metric_values.shape[-1] != len(EPDMS_METRICS):
        raise ValueError(
            f"Expected {len(EPDMS_METRICS)} EPDMS metrics, got "
            f"shape {tuple(metric_values.shape)}"
        )
    if eps <= 0:
        raise ValueError("eps must be positive")

    values = metric_values.clamp(float(eps), 1.0)
    multiplicative_log_score = (
        values[..., :MULTIPLICATIVE_METRIC_COUNT].log().sum(dim=-1)
    )
    # DrivoR/official weights: EP=5, TTC=5, LK=2, HC=2.
    stage1_weighted_values = values[
        ..., MULTIPLICATIVE_METRIC_COUNT:EXTENDED_COMFORT_INDEX
    ]
    stage1_weights = values.new_tensor(WEIGHTED_METRIC_WEIGHTS[:-1])
    utility_log_score = (
        (stage1_weighted_values * stage1_weights)
        .sum(dim=-1)
        .clamp_min(float(eps))
        .log()
    )
    return multiplicative_log_score + utility_log_score


def compose_epdms(
    metric_values: torch.Tensor,
    *,
    include_extended_comfort: bool = True,
) -> torch.Tensor:
    """Compose metric probabilities/regressions with the official v2 formula.

    Args:
        metric_values: tensor with final dimension ``len(EPDMS_METRICS)``.
        include_extended_comfort: false computes the stage-1-only score used by
            the official scorer before cross-frame extended comfort is known.
    """

    if metric_values.shape[-1] != len(EPDMS_METRICS):
        raise ValueError(
            f"Expected {len(EPDMS_METRICS)} EPDMS metrics, got "
            f"shape {tuple(metric_values.shape)}"
        )

    values = metric_values.clamp(0.0, 1.0)
    multiplicative = values[..., :MULTIPLICATIVE_METRIC_COUNT].prod(dim=-1)
    weighted = values[..., MULTIPLICATIVE_METRIC_COUNT:]
    weights = weighted.new_tensor(WEIGHTED_METRIC_WEIGHTS)
    if not include_extended_comfort:
        weighted = weighted[..., :-1]
        weights = weights[:-1]
    weighted_mean = (weighted * weights).sum(dim=-1) / weights.sum()
    return multiplicative * weighted_mean


def targets_from_official_score_array(
    official_scores: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert the local official scorer array into canonical nine-metric order.

    ``compute_navsim_score.after_score_proposals`` emits
    ``NC,DAC,TLC,DDC,EP,TTC,LK,HC,stage1_score``.  Extended comfort is a
    cross-frame metric and is therefore marked invalid rather than fabricated.
    """

    if official_scores.shape[-1] != 9:
        raise ValueError(
            "Expected official score array [NC,DAC,TLC,DDC,EP,TTC,LK,HC,score]"
        )

    target = official_scores.new_full((*official_scores.shape[:-1], 9), float("nan"))
    target[..., 0] = official_scores[..., 0]
    target[..., 1] = official_scores[..., 1]
    target[..., 2] = official_scores[..., 3]
    target[..., 3] = official_scores[..., 2]
    target[..., 4:8] = official_scores[..., 4:8]
    valid = torch.isfinite(target)
    return target, valid, official_scores[..., 8]


def masked_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCE supporting unavailable targets such as per-proposal EC."""

    if valid_mask is None:
        valid_mask = torch.isfinite(targets)
    if not torch.any(valid_mask):
        return logits.sum() * 0.0
    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
    loss = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
    return loss[valid_mask].mean()


@dataclass(frozen=True)
class RankingResult:
    loss: torch.Tensor
    category_losses: Mapping[str, torch.Tensor]
    pair_counts: Mapping[str, int]


RANKING_CATEGORIES: Tuple[str, ...] = (
    "elite_vs_high_safe",
    "elite_vs_good_safe",
    "safe_vs_violation",
    "other_return_gap",
)
DEFAULT_CATEGORY_WEIGHTS: Tuple[float, ...] = (0.45, 0.20, 0.25, 0.10)


def _return_bins(returns: torch.Tensor, quantile_edges: Sequence[float]) -> torch.Tensor:
    if len(quantile_edges) != 4:
        raise ValueError("quantile_edges must contain calibrated q20,q50,q80,q95")
    edges = returns.new_tensor(tuple(float(v) for v in quantile_edges))
    if not torch.all(edges[1:] >= edges[:-1]):
        raise ValueError("quantile_edges must be non-decreasing")
    bins = torch.zeros_like(returns, dtype=torch.long)
    bins[returns > edges[0]] = 1
    bins[returns >= edges[1]] = 2
    bins[returns >= edges[2]] = 3
    bins[returns >= edges[3]] = 4
    return bins


def stratified_generated_ranking_loss(
    predicted_scores: torch.Tensor,
    true_returns: torch.Tensor,
    metric_targets: torch.Tensor,
    *,
    quantile_edges: Sequence[float],
    min_return_gap: float,
    margin: float = 0.02,
    category_weights: Sequence[float] = DEFAULT_CATEGORY_WEIGHTS,
) -> RankingResult:
    """Compute the v2 generated-proposal stratified pairwise ranking loss.

    Category losses are means and active category weights are renormalized.  A
    compliance pair always ranks the fully compliant proposal first, independent
    of the scalar return, matching the safety rule in the migration prompt.
    """

    if predicted_scores.ndim != 2 or true_returns.shape != predicted_scores.shape:
        raise ValueError("predicted_scores and true_returns must have shape [B,K]")
    if metric_targets.shape != (*predicted_scores.shape, len(EPDMS_METRICS)):
        raise ValueError("metric_targets must have shape [B,K,9]")
    if len(category_weights) != len(RANKING_CATEGORIES):
        raise ValueError("category_weights must contain four values")
    if any(float(weight) < 0 for weight in category_weights):
        raise ValueError("category_weights must be non-negative")

    bins = _return_bins(true_returns, quantile_edges)
    compliant = torch.all(metric_targets[..., :4] >= (1.0 - 1e-6), dim=-1)
    finite = torch.isfinite(true_returns) & torch.isfinite(predicted_scores)

    category_values: Dict[str, list[torch.Tensor]] = {name: [] for name in RANKING_CATEGORIES}
    for batch_idx in range(predicted_scores.shape[0]):
        score = predicted_scores[batch_idx]
        ret = true_returns[batch_idx]
        safe = compliant[batch_idx]
        valid = finite[batch_idx]
        ret_gap = ret[:, None] - ret[None, :]

        elite = bins[batch_idx] == 4
        high = bins[batch_idx] == 3
        good = bins[batch_idx] == 2
        pair_valid = valid[:, None] & valid[None, :]

        masks = {
            "elite_vs_high_safe": (
                elite[:, None] & high[None, :] & safe[:, None] & safe[None, :] & pair_valid
            ),
            "elite_vs_good_safe": (
                elite[:, None] & good[None, :] & safe[:, None] & safe[None, :] & pair_valid
            ),
            "safe_vs_violation": safe[:, None] & (~safe[None, :]) & pair_valid,
        }
        used = masks["elite_vs_high_safe"] | masks["elite_vs_good_safe"] | masks["safe_vs_violation"]
        masks["other_return_gap"] = (
            (ret_gap >= float(min_return_gap)) & pair_valid & (~used)
        )

        score_gap = score[:, None] - score[None, :]
        for name, mask in masks.items():
            if torch.any(mask):
                category_values[name].append(F.softplus(float(margin) - score_gap[mask]))

    category_losses: Dict[str, torch.Tensor] = {}
    pair_counts: Dict[str, int] = {}
    active_losses = []
    active_weights = []
    for name, weight in zip(RANKING_CATEGORIES, category_weights):
        values = category_values[name]
        count = sum(value.numel() for value in values)
        pair_counts[name] = count
        if count:
            loss = torch.cat(values).mean()
            category_losses[name] = loss
            active_losses.append(loss)
            active_weights.append(float(weight))

    if not active_losses:
        zero = predicted_scores.sum() * 0.0
        return RankingResult(zero, category_losses, pair_counts)

    normalized_weights = predicted_scores.new_tensor(active_weights)
    if normalized_weights.sum() <= 0:
        raise ValueError("active ranking category weights sum to zero")
    normalized_weights = normalized_weights / normalized_weights.sum()
    total = torch.stack(active_losses).mul(normalized_weights).sum()
    return RankingResult(total, category_losses, pair_counts)
