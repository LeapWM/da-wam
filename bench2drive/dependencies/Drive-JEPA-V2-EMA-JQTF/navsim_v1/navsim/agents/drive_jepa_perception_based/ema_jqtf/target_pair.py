"""Camera-pair contract for the EMA future target."""

from __future__ import annotations

import torch


FUTURE_TARGET_PAIR_MODES = ("repeat_future", "current_future")


def build_future_target_camera_pair(
    current_frame: torch.Tensor,
    future_frame: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Build the two-frame clip consumed by the EMA target encoder.

    Args:
        current_frame: Normalized frame at ``t``, shaped ``[B,3,H,W]``.
        future_frame: Normalized frame at ``t+0.5``, shaped ``[B,3,H,W]``.
        mode: ``current_future`` for ``[t,t+0.5]`` or ``repeat_future`` for
            the historical ``[t+0.5,t+0.5]`` target.
    """
    if mode not in FUTURE_TARGET_PAIR_MODES:
        raise ValueError(
            f"unsupported future target pair mode {mode!r}; "
            f"expected one of {FUTURE_TARGET_PAIR_MODES}"
        )
    if current_frame.shape != future_frame.shape:
        raise ValueError(
            "current and future frames must have identical shapes, got "
            f"{tuple(current_frame.shape)} and {tuple(future_frame.shape)}"
        )
    if current_frame.ndim != 4:
        raise ValueError(
            "camera frames must be [B,3,H,W], got "
            f"{tuple(current_frame.shape)}"
        )
    if mode == "current_future":
        return torch.stack((current_frame, future_frame), dim=1)
    return torch.stack((future_frame, future_frame), dim=1)
