"""Camera-pair contract for the EMA future target."""

import torch


def build_current_future_pair(
    current_frame: torch.Tensor,
    future_frame: torch.Tensor,
) -> torch.Tensor:
    """Return the two-frame clip ``[t, t+0.5]`` consumed by V-JEPA."""
    if current_frame.ndim != 4 or future_frame.ndim != 4:
        raise ValueError("camera frames must be [B,3,H,W]")
    if current_frame.shape != future_frame.shape:
        raise ValueError(
            "current and future frames must have identical shapes, got "
            f"{tuple(current_frame.shape)} and {tuple(future_frame.shape)}"
        )
    return torch.stack((current_frame, future_frame), dim=1)
