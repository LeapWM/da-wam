"""LoRA adapters for V-JEPA 2.0 / 2.1 image encoder (adapted from DrivoR)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _LoRA_qkv_vjepa(nn.Module):
    """Replace V-JEPA block's qkv linear with Q/V LoRA adapters."""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class LoRA_VJEPA(nn.Module):
    """Wrap a V-JEPA vit_model, freeze it, and insert LoRA adapters on Q/V of selected blocks.

    Args:
        vit_model: Loaded V-JEPA vision transformer (v2.0 or v2.1).
        r: LoRA rank. If 0, no adapters are added (fully frozen).
        lora_layer: List of block indices to apply LoRA. None = all blocks.
    """

    def __init__(self, vit_model: nn.Module, r: int, lora_layer: list[int] | None = None):
        super().__init__()
        if r == 0:
            for param in vit_model.parameters():
                param.requires_grad = False
            self.lora_vit = vit_model
            self.w_As: list[nn.Linear] = []
            self.w_Bs: list[nn.Linear] = []
            return

        self.lora_layer = lora_layer if lora_layer is not None else list(range(len(vit_model.blocks)))

        self.w_As: list[nn.Linear] = []
        self.w_Bs: list[nn.Linear] = []

        # Freeze all original params
        for param in vit_model.parameters():
            param.requires_grad = False

        # Insert LoRA adapters on selected blocks
        for t_layer_i, blk in enumerate(vit_model.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.extend([w_a_linear_q, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q, w_b_linear_v])
            blk.attn.qkv = _LoRA_qkv_vjepa(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.reset_parameters()
        self.lora_vit = vit_model

    def reset_parameters(self) -> None:
        for w_a in self.w_As:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_Bs:
            nn.init.zeros_(w_b.weight)

    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.lora_vit.parameters() if p.requires_grad)
