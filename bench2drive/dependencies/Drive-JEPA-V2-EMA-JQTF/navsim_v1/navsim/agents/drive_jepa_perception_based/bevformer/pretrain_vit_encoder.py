"""Frozen pretrain V-JEPA encoder for future latent prediction (FL-2)."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from navsim.agents.drive_jepa_perception_free.vjepa_encoder import load_vjepa_encoder
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig

VIT_FEATURE_HW = (16, 32)


def encode_vit_spatial_map(vit_model: nn.Module, img: torch.Tensor) -> torch.Tensor:
    """Encode multi-frame images to a spatial vit feature map.

    Args:
        vit_model: V-JEPA vit (no LoRA, no projector).
        img: [B, N, 3, H, W], ImageNet-normalized.
    Returns:
        [B, C, H, W] with H=16, W=32 for 256x512 input.
    """
    batch_size, num_frames, channels, height, width = img.shape
    x = img.reshape(batch_size * num_frames, channels, height, width)
    x = rearrange(x, "(B N) C H W -> B C N H W", B=batch_size)
    tokens = vit_model(x)
    feat_h, feat_w = VIT_FEATURE_HW
    return rearrange(tokens, "B (H W) C -> B C H W", H=feat_h, W=feat_w)


class FrozenPretrainVitEncoder(nn.Module):
    """Pretrain V-JEPA vit only; weights are rebuilt from pretrain_pt_path (not saved in ckpt)."""

    def __init__(self, config: DriveJEPAConfig):
        super().__init__()
        vit = load_vjepa_encoder(
            vjepa_version=config.vjepa_version,
            resolution=(256, 512),
            checkpoint=config.pretrain_pt_path,
            image_architecture=config.image_architecture,
            num_frames=2,
            register_prehook=False,
        )
        for param in vit.parameters():
            param.requires_grad = False
        # Keep vit out of Module._modules so ckpt load/save skips pretrain weights.
        object.__setattr__(self, "vit", vit)

    def _apply(self, fn):
        super()._apply(fn)
        self.vit = self.vit._apply(fn)
        return self

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return encode_vit_spatial_map(self.vit, img)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
        return self
