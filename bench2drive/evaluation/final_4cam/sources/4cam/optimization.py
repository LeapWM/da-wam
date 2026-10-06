"""Optimizer schedule shared with the local SparseDrive-style recipe."""
import math

import torch


def lr_factor(step, total_steps, warmup_steps, min_ratio):
    """Linear 1/3 -> 1 warmup followed by cosine decay."""
    warmup = min(warmup_steps, max(0, total_steps - 1))
    if warmup and step < warmup:
        return 1 / 3 + (2 / 3) * step / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup - 1)))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def grouped_adamw(groups, total_steps, warmup_steps=500, weight_decay=1e-3, min_lr=1e-6):
    """Build AdamW without decaying biases/norm scales, preserving group LRs."""
    if total_steps < 1 or warmup_steps < 0 or weight_decay < 0:
        raise ValueError("Invalid optimizer schedule")
    optimizer_groups = []
    trainable_lrs = [float(group["lr"]) for group in groups if any(p.requires_grad for _, p in group["named_params"])]
    if not trainable_lrs:
        raise ValueError("No trainable optimizer parameters")
    base_lr = min(trainable_lrs)
    if not 0 < min_lr <= base_lr:
        raise ValueError(f"min_lr={min_lr} must be in (0, base lr={base_lr}]")
    for group in groups:
        lr = float(group["lr"])
        decay, no_decay = [], []
        for name, parameter in group["named_params"]:
            if not parameter.requires_grad:
                continue
            (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
        common = {"lr": lr, "initial_lr": lr, "source_group": group["name"]}
        if decay:
            optimizer_groups.append(dict(common, params=decay, weight_decay=weight_decay, decay=True))
        if no_decay:
            optimizer_groups.append(dict(common, params=no_decay, weight_decay=0.0, decay=False))
    optimizer = torch.optim.AdamW(optimizer_groups)
    # One multiplicative curve keeps the configured group ratios intact.
    lambdas = [lambda step: lr_factor(step, total_steps, warmup_steps, min_lr / base_lr)] * len(optimizer_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }
