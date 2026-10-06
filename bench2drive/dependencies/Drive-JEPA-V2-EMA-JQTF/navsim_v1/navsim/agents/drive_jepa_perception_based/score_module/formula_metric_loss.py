"""Cost-sensitive metric loss for Formula-progress scorer heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalized_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def formula_metric_bce_per_head(
    metric_logits: torch.Tensor,
    metric_target: torch.Tensor,
    *,
    noc_failure_weight: float = 1.0,
    noc_partial_weight: float = 1.0,
    ttc_failure_weight: float = 1.0,
) -> torch.Tensor:
    """Return normalized BCE losses in [NOC, DAC, TTC, Comfort] order.

    Formula targets use NOC in {0, 0.5, 1} and TTC in {0, 1}. Failure
    weighting is normalized independently per head, changing class cost
    without changing the head's nominal scale.
    """
    if metric_logits.shape != metric_target.shape:
        raise ValueError(
            "Formula metric logits and targets must have the same shape, got "
            f"{tuple(metric_logits.shape)} vs {tuple(metric_target.shape)}"
        )
    if metric_logits.ndim < 2 or metric_logits.shape[-1] != 4:
        raise ValueError(
            "Formula metric tensors must end in four heads, got "
            f"{tuple(metric_logits.shape)}"
        )
    weights = (
        noc_failure_weight,
        noc_partial_weight,
        ttc_failure_weight,
    )
    if any(float(weight) <= 0.0 for weight in weights):
        raise ValueError("Formula metric class weights must be positive")

    element = F.binary_cross_entropy_with_logits(
        metric_logits.float(),
        metric_target.float(),
        reduction="none",
    )
    noc_target = metric_target[..., 0].float()
    noc_weights = torch.ones_like(noc_target)
    noc_weights = torch.where(
        noc_target <= 0.05,
        torch.full_like(noc_weights, float(noc_failure_weight)),
        noc_weights,
    )
    noc_weights = torch.where(
        (noc_target > 0.05) & (noc_target < 0.95),
        torch.full_like(noc_weights, float(noc_partial_weight)),
        noc_weights,
    )
    ttc_target = metric_target[..., 2].float()
    ttc_weights = torch.where(
        ttc_target < 0.5,
        torch.full_like(ttc_target, float(ttc_failure_weight)),
        torch.ones_like(ttc_target),
    )

    reduce_dims = tuple(range(element.ndim - 1))
    return torch.stack(
        (
            _normalized_weighted_mean(element[..., 0], noc_weights),
            element[..., 1].mean(dim=reduce_dims),
            _normalized_weighted_mean(element[..., 2], ttc_weights),
            element[..., 3].mean(dim=reduce_dims),
        )
    )


def formula_metric_focal_per_head(
    metric_logits: torch.Tensor,
    metric_target: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
    noc_failure_weight: float = 1.0,
    noc_partial_weight: float = 1.0,
    ttc_failure_weight: float = 1.0,
) -> torch.Tensor:
    """Return sigmoid Focal losses for all four Formula metric heads.

    ``alpha`` is the weight of a safe/positive target; failures receive
    ``1 - alpha``. Optional NOC/TTC weights are composed with alpha and then
    normalized independently per head, so they change sample cost without an
    arbitrary global loss-scale change. Soft NOC targets such as 0.5 retain
    their cached value instead of being thresholded into a class.
    """
    if metric_logits.shape != metric_target.shape:
        raise ValueError(
            "Formula metric logits and targets must have the same shape, got "
            f"{tuple(metric_logits.shape)} vs {tuple(metric_target.shape)}"
        )
    if metric_logits.ndim < 2 or metric_logits.shape[-1] != 4:
        raise ValueError(
            "Formula metric tensors must end in four heads, got "
            f"{tuple(metric_logits.shape)}"
        )
    if float(gamma) < 0.0:
        raise ValueError("Formula Focal gamma must be non-negative")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("Formula Focal alpha must be within (0,1)")
    class_weights = (
        noc_failure_weight,
        noc_partial_weight,
        ttc_failure_weight,
    )
    if any(float(weight) <= 0.0 for weight in class_weights):
        raise ValueError("Formula Focal class weights must be positive")

    logits = metric_logits.float()
    targets = metric_target.float()
    probabilities = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    focal_factor = (1.0 - p_t).pow(float(gamma))
    alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
    element = bce * focal_factor

    external_weight = torch.ones_like(element)
    noc_target = targets[..., 0]
    external_weight[..., 0] = torch.where(
        noc_target <= 0.05,
        torch.full_like(noc_target, float(noc_failure_weight)),
        external_weight[..., 0],
    )
    external_weight[..., 0] = torch.where(
        (noc_target > 0.05) & (noc_target < 0.95),
        torch.full_like(noc_target, float(noc_partial_weight)),
        external_weight[..., 0],
    )
    ttc_target = targets[..., 2]
    external_weight[..., 2] = torch.where(
        ttc_target < 0.5,
        torch.full_like(ttc_target, float(ttc_failure_weight)),
        external_weight[..., 2],
    )
    combined_weight = alpha_t * external_weight
    return torch.stack(
        tuple(
            _normalized_weighted_mean(
                element[..., head], combined_weight[..., head]
            )
            for head in range(4)
        )
    )
