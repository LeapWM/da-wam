"""Small epoch-indexed loss-weight schedules for continuation training."""

from __future__ import annotations

import math
from typing import Iterable


def scheduled_loss_weight(
    schedule: Iterable[float],
    epoch: int,
    default: float,
) -> float:
    """Return an epoch-indexed weight, holding the last value afterwards."""
    values = tuple(float(value) for value in schedule)
    default = float(default)
    if not math.isfinite(default) or default < 0.0:
        raise ValueError("default loss weight must be finite and non-negative")
    if not values:
        return default
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("scheduled loss weights must be finite and non-negative")
    index = min(max(int(epoch), 0), len(values) - 1)
    return values[index]


def bundled_loss_weight(
    base_weight: float,
    scorer_weight: float,
    enabled: bool,
) -> float:
    """Optionally scale an auxiliary loss with the scorer bundle."""
    base_weight = float(base_weight)
    scorer_weight = float(scorer_weight)
    if not math.isfinite(base_weight) or base_weight < 0.0:
        raise ValueError("base loss weight must be finite and non-negative")
    if not math.isfinite(scorer_weight) or scorer_weight < 0.0:
        raise ValueError("scorer loss weight must be finite and non-negative")
    return base_weight * scorer_weight if enabled else base_weight
