"""Cost-sensitive final-score loss for safety-critical scorer mistakes."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AsymmetricSafetyLossOutput:
    total: torch.Tensor
    false_safe: torch.Tensor
    false_unsafe: torch.Tensor
    false_safe_rate: torch.Tensor
    false_unsafe_rate: torch.Tensor
    unsafe_fraction: torch.Tensor
    safe_high_fraction: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=values.dtype)
    return (values * mask_float).sum() / mask_float.sum().clamp(min=1.0)


def asymmetric_safety_loss(
    final_logits: torch.Tensor,
    target_scores: torch.Tensor,
    *,
    false_safe_weight: float = 3.0,
    false_unsafe_weight: float = 1.0,
    error_margin: float = 0.05,
    safety_metric_threshold: float = 0.95,
    safe_final_threshold: float = 0.9,
) -> AsymmetricSafetyLossOutput:
    """Penalize unsafe overprediction more than high-quality safe underprediction.

    ``target_scores`` must end with the NAVSIM score columns
    ``[NOC, DAC, EP, TTC, comfort, final]``.  Safety follows the established
    diagnostic definition ``NOC >= threshold and TTC >= threshold``.  DAC
    remains represented in the final-score target but is not treated as a
    collision-safety label here.

    ``final_logits`` predicts the continuous final score, not a separate safety
    probability.  Therefore errors are directional residuals relative to the
    final target rather than a second binary-classification objective.
    """

    if final_logits.shape != target_scores.shape[:-1]:
        raise ValueError(
            "final_logits must align with target_scores prefix, got "
            f"{tuple(final_logits.shape)} vs {tuple(target_scores.shape)}"
        )
    if target_scores.shape[-1] < 6:
        raise ValueError(
            "target_scores must contain trailing "
            "[NOC,DAC,EP,TTC,comfort,final] columns"
        )
    if false_safe_weight < 0 or false_unsafe_weight < 0:
        raise ValueError("asymmetric safety class weights must be non-negative")
    if error_margin < 0:
        raise ValueError("asymmetric safety error_margin must be non-negative")
    if not 0 <= safety_metric_threshold <= 1:
        raise ValueError("safety_metric_threshold must be in [0,1]")
    if not 0 <= safe_final_threshold <= 1:
        raise ValueError("safe_final_threshold must be in [0,1]")

    prediction = torch.sigmoid(final_logits.float())
    scores = target_scores.float()
    target_final = scores[..., -1]
    noc = scores[..., -6]
    ttc = scores[..., -3]

    safe_mask = (noc >= safety_metric_threshold) & (
        ttc >= safety_metric_threshold
    )
    unsafe_mask = ~safe_mask
    safe_high_mask = safe_mask & (target_final >= safe_final_threshold)

    false_safe_error = F.relu(
        prediction - target_final - error_margin
    ).square()
    false_unsafe_error = F.relu(
        target_final - prediction - error_margin
    ).square()

    false_safe = _masked_mean(false_safe_error, unsafe_mask)
    false_unsafe = _masked_mean(false_unsafe_error, safe_high_mask)
    total = (
        float(false_safe_weight) * false_safe
        + float(false_unsafe_weight) * false_unsafe
    )

    false_safe_rate = _masked_mean(
        (false_safe_error > 0).to(prediction.dtype), unsafe_mask
    )
    false_unsafe_rate = _masked_mean(
        (false_unsafe_error > 0).to(prediction.dtype), safe_high_mask
    )

    return AsymmetricSafetyLossOutput(
        total=total,
        false_safe=false_safe,
        false_unsafe=false_unsafe,
        false_safe_rate=false_safe_rate.detach(),
        false_unsafe_rate=false_unsafe_rate.detach(),
        unsafe_fraction=unsafe_mask.float().mean().detach(),
        safe_high_fraction=safe_high_mask.float().mean().detach(),
    )
