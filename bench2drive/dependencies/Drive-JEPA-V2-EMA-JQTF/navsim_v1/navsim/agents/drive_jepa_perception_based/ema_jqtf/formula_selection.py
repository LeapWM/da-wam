"""Shared Formula-progress composition for training and inference."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


FORMULA_SELECTION_MODES = (
    "current",
    "exact_pdm",
    "safety_first",
)
FORMULA_PROGRESS_MODES = (
    "symmetric",
    "official_threshold",
    "one_sided_no5m",
)

# Keep the same numerical stabilizer as the offline scorer's metre-space GT.
FORMULA_PROGRESS_EPSILON_METERS = 0.01


class FormulaSelectionOutput(NamedTuple):
    formula_score: torch.Tensor
    selection_score: torch.Tensor
    ego_progress: torch.Tensor
    safety_rank: torch.Tensor


def formula_progress_ratio(
    raw_progress,
    pdm_progress,
    *,
    epsilon: float = FORMULA_PROGRESS_EPSILON_METERS,
):
    """Return the legacy symmetric candidate/reference progress ratio.

    This function deliberately supports both NumPy labels and torch model
    outputs. A stopped candidate receives full progress only when the
    reference is also stopped; there is no low-distance full-score branch.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    raw_is_tensor = torch.is_tensor(raw_progress)
    pdm_is_tensor = torch.is_tensor(pdm_progress)
    if raw_is_tensor != pdm_is_tensor:
        raise TypeError(
            "raw_progress and pdm_progress must both be torch tensors or "
            "both be NumPy-compatible values"
        )

    if raw_is_tensor:
        raw = raw_progress.clamp_min(0.0)
        reference = pdm_progress.clamp_min(0.0)
        minimum = torch.minimum(raw, reference)
        maximum = torch.maximum(raw, reference)
        return ((minimum + epsilon) / (maximum + epsilon)).clamp(0.0, 1.0)

    raw = np.clip(np.asarray(raw_progress), a_min=0.0, a_max=None)
    reference = np.clip(np.asarray(pdm_progress), a_min=0.0, a_max=None)
    minimum = np.minimum(raw, reference)
    maximum = np.maximum(raw, reference)
    return np.clip((minimum + epsilon) / (maximum + epsilon), 0.0, 1.0)


def official_threshold_progress(
    raw_progress,
    pdm_progress,
    multiplicative_safety,
    *,
    distance_threshold: float = 5.0,
    epsilon: float = FORMULA_PROGRESS_EPSILON_METERS,
):
    """Differentiable surrogate of official pairwise NAVSIM Progress.

    With binary safety labels this exactly matches the official rule: grant
    full Progress below the distance threshold, otherwise normalize gated
    candidate distance by ``max(candidate, PDM baseline)``. Continuous safety
    probabilities replace the hard safe/unsafe branch during model inference.
    """
    if distance_threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    raw_is_tensor = torch.is_tensor(raw_progress)
    if (
        raw_is_tensor != torch.is_tensor(pdm_progress)
        or raw_is_tensor != torch.is_tensor(multiplicative_safety)
    ):
        raise TypeError(
            "raw, PDM progress, and safety must all be torch tensors or all "
            "be NumPy-compatible values"
        )

    if raw_is_tensor:
        raw = raw_progress.clamp_min(0.0)
        reference = pdm_progress.clamp_min(0.0)
        safety = multiplicative_safety.clamp(0.0, 1.0)
        gated_raw = raw * safety
        maximum = torch.maximum(gated_raw, reference)
        normalized = gated_raw / maximum.clamp_min(float(epsilon))
        return torch.where(
            maximum > float(distance_threshold), normalized, safety
        ).clamp(0.0, 1.0)

    raw = np.clip(np.asarray(raw_progress), a_min=0.0, a_max=None)
    reference = np.clip(np.asarray(pdm_progress), a_min=0.0, a_max=None)
    safety = np.clip(np.asarray(multiplicative_safety), 0.0, 1.0)
    gated_raw = raw * safety
    maximum = np.maximum(gated_raw, reference)
    normalized = gated_raw / np.maximum(maximum, float(epsilon))
    return np.clip(
        np.where(maximum > float(distance_threshold), normalized, safety),
        0.0,
        1.0,
    )


def one_sided_progress(
    raw_progress,
    pdm_progress,
    multiplicative_safety,
    *,
    epsilon: float = FORMULA_PROGRESS_EPSILON_METERS,
):
    """Official-style one-sided Progress without the five-metre branch.

    A safe candidate below the PDM baseline receives ``candidate / PDM``;
    matching or exceeding the baseline saturates at one.  This never penalizes
    a candidate merely for travelling farther than the baseline.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    raw_is_tensor = torch.is_tensor(raw_progress)
    if (
        raw_is_tensor != torch.is_tensor(pdm_progress)
        or raw_is_tensor != torch.is_tensor(multiplicative_safety)
    ):
        raise TypeError(
            "raw, PDM progress, and safety must all be torch tensors or all "
            "be NumPy-compatible values"
        )

    if raw_is_tensor:
        raw = raw_progress.clamp_min(0.0)
        reference = pdm_progress.clamp_min(0.0)
        safety = multiplicative_safety.clamp(0.0, 1.0)
        gated_raw = raw * safety
        maximum = torch.maximum(gated_raw, reference)
        return (gated_raw / maximum.clamp_min(float(epsilon))).clamp(
            0.0, 1.0
        )

    raw = np.clip(np.asarray(raw_progress), a_min=0.0, a_max=None)
    reference = np.clip(np.asarray(pdm_progress), a_min=0.0, a_max=None)
    safety = np.clip(np.asarray(multiplicative_safety), 0.0, 1.0)
    gated_raw = raw * safety
    maximum = np.maximum(gated_raw, reference)
    return np.clip(
        gated_raw / np.maximum(maximum, float(epsilon)), 0.0, 1.0
    )


def formula_progress_selection(
    metric_scores: torch.Tensor,
    raw_progress_norm: torch.Tensor,
    pdm_progress_norm: torch.Tensor,
    *,
    progress_scale: float,
    mode: str = "current",
    safety_topk: int = 8,
    progress_mode: str = "symmetric",
    progress_distance_threshold: float = 5.0,
    safe_progress_tie_break_weight: float = 0.0,
) -> FormulaSelectionOutput:
    """Compose the GT-aligned Formula score used in training and inference.

    ``current`` and the legacy ``exact_pdm`` alias share the selected Progress
    composition. ``safety_first`` changes only the inference shortlist; it
    does not change the Formula score itself.
    """
    if mode not in FORMULA_SELECTION_MODES:
        raise ValueError(
            f"mode must be one of {FORMULA_SELECTION_MODES}, got {mode!r}"
        )
    if progress_mode not in FORMULA_PROGRESS_MODES:
        raise ValueError(
            f"progress_mode must be one of {FORMULA_PROGRESS_MODES}, got "
            f"{progress_mode!r}"
        )
    if metric_scores.ndim != 3 or metric_scores.shape[-1] != 4:
        raise ValueError(
            "metric_scores must be [B,P,4], got "
            f"{tuple(metric_scores.shape)}"
        )
    if raw_progress_norm.shape != metric_scores.shape[:-1]:
        raise ValueError(
            "raw_progress_norm must be [B,P], got "
            f"{tuple(raw_progress_norm.shape)}"
        )
    if pdm_progress_norm.shape not in (
        (metric_scores.shape[0], 1),
        tuple(raw_progress_norm.shape),
    ):
        raise ValueError(
            "pdm_progress_norm must be [B,1] or [B,P], got "
            f"{tuple(pdm_progress_norm.shape)}"
        )
    if progress_scale <= 0.0:
        raise ValueError("progress_scale must be positive")
    if safety_topk <= 0:
        raise ValueError("safety_topk must be positive")
    if safe_progress_tie_break_weight < 0.0:
        raise ValueError("safe_progress_tie_break_weight must be non-negative")

    noc, dac, ttc, comfort = metric_scores.unbind(dim=-1)
    pdm_progress_norm = pdm_progress_norm.expand_as(raw_progress_norm)
    multiplicative_safety = noc * dac
    if progress_mode == "official_threshold":
        ego_progress = official_threshold_progress(
            raw_progress_norm,
            pdm_progress_norm,
            multiplicative_safety,
            distance_threshold=(
                float(progress_distance_threshold) / progress_scale
            ),
            epsilon=FORMULA_PROGRESS_EPSILON_METERS / progress_scale,
        )
    elif progress_mode == "one_sided_no5m":
        ego_progress = one_sided_progress(
            raw_progress_norm,
            pdm_progress_norm,
            multiplicative_safety,
            epsilon=FORMULA_PROGRESS_EPSILON_METERS / progress_scale,
        )
    else:
        progress_ratio = formula_progress_ratio(
            raw_progress_norm,
            pdm_progress_norm,
            epsilon=FORMULA_PROGRESS_EPSILON_METERS / progress_scale,
        )
        ego_progress = multiplicative_safety * progress_ratio

    formula_score = (
        multiplicative_safety
        * (5.0 * ttc + 5.0 * ego_progress + 2.0 * comfort)
        / 12.0
    )
    safety_rank = multiplicative_safety * ttc
    # Keep the reported Formula score exactly official-like.  This bounded
    # auxiliary affects candidate selection only and resolves EP ties in
    # favour of farther GT-safe trajectories learned by the raw-progress rank.
    threshold_norm = float(progress_distance_threshold) / progress_scale
    raw_for_tie_break = raw_progress_norm.clamp_min(0.0)
    bounded_raw_progress = raw_for_tie_break / (
        raw_for_tie_break
        + max(
            threshold_norm,
            FORMULA_PROGRESS_EPSILON_METERS / progress_scale,
        )
    )
    selection_score = formula_score + (
        float(safe_progress_tie_break_weight)
        * safety_rank
        * bounded_raw_progress
    )

    if mode == "safety_first":
        topk = min(int(safety_topk), formula_score.shape[1])
        eligible_index = torch.topk(
            safety_rank,
            k=topk,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        eligible = torch.zeros_like(formula_score, dtype=torch.bool)
        eligible.scatter_(1, eligible_index, True)
        selection_score = selection_score.masked_fill(~eligible, -1.0)

    return FormulaSelectionOutput(
        formula_score=formula_score,
        selection_score=selection_score,
        ego_progress=ego_progress,
        safety_rank=safety_rank,
    )
