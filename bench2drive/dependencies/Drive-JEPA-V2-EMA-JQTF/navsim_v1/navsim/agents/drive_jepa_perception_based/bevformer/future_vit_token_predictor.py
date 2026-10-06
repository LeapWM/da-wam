"""Small JEPA-style token predictor for frozen pretrain vit future latents."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from navsim.agents.drive_jepa_perception_based.bevformer.pretrain_vit_encoder import VIT_FEATURE_HW


def spatial_map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, H*W, C]."""
    return rearrange(feature_map, "b c h w -> b (h w) c")


def tokens_to_spatial_map(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[B, H*W, C] -> [B, C, H, W]."""
    return rearrange(tokens, "b (h w) c -> b c h w", h=height, w=width)


class FutureVitTokenPredictor(nn.Module):
    """JEPA predictor with optional GT-latent and proposal-rollout conditioning."""

    def __init__(
        self,
        token_dim: int = 1024,
        num_offsets: int = 1,
        spatial_hw=VIT_FEATURE_HW,
        predictor_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        trajectory_conditioned: bool = False,
        trajectory_dim: int = 24,
        rollout_dim: int = 256,
        rollout_depth: int = 2,
    ):
        super().__init__()
        self.num_offsets = num_offsets
        self.token_dim = token_dim
        self.predictor_dim = predictor_dim
        feat_h, feat_w = spatial_hw
        self.num_patches = feat_h * feat_w
        self.spatial_hw = spatial_hw
        self.trajectory_conditioned = trajectory_conditioned

        self.context_embed = nn.Linear(token_dim, predictor_dim, bias=True)
        self.context_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, predictor_dim))
        self.future_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, predictor_dim))

        # One learnable future query bank per offset (JEPA mask-token style).
        self.future_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.num_patches, predictor_dim))
            for _ in range(num_offsets)
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_dim * mlp_ratio),
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.predictor_norm = nn.LayerNorm(predictor_dim)
        self.predictor_proj = nn.Linear(predictor_dim, token_dim, bias=True)

        # The full latent is supervised only for the observed GT trajectory.
        # Counterfactual proposals use compact rollout features trained only by
        # scorer metric/final/ranking losses, never by the GT latent target.
        if trajectory_conditioned:
            self.trajectory_embed = nn.Sequential(
                nn.Linear(trajectory_dim, predictor_dim),
                nn.GELU(),
                nn.Linear(predictor_dim, predictor_dim),
            )
            # Keep candidate score gradients out of the GT full-latent head.
            self.rollout_context_embed = nn.Linear(token_dim, predictor_dim, bias=True)
            self.rollout_context_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, predictor_dim)
            )
            self.rollout_trajectory_embed = nn.Sequential(
                nn.Linear(trajectory_dim, predictor_dim),
                nn.GELU(),
                nn.Linear(predictor_dim, predictor_dim),
            )
            rollout_layer = nn.TransformerDecoderLayer(
                d_model=predictor_dim,
                nhead=num_heads,
                dim_feedforward=int(predictor_dim * mlp_ratio),
                dropout=drop,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.rollout_blocks = nn.TransformerDecoder(
                rollout_layer, num_layers=rollout_depth
            )
            self.rollout_norm = nn.LayerNorm(predictor_dim)
            self.rollout_proj = nn.Linear(predictor_dim, rollout_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.context_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.future_pos_embed, std=0.02)
        for query in self.future_queries:
            nn.init.trunc_normal_(query, std=0.02)
        if self.trajectory_conditioned:
            nn.init.trunc_normal_(self.rollout_context_pos_embed, std=0.02)

    def _context_tokens(self, current_latent: torch.Tensor) -> torch.Tensor:
        context_tokens = spatial_map_to_tokens(current_latent)
        return self.context_embed(context_tokens) + self.context_pos_embed

    def _rollout_context_tokens(self, current_latent: torch.Tensor) -> torch.Tensor:
        context_tokens = spatial_map_to_tokens(current_latent)
        return self.rollout_context_embed(context_tokens) + self.rollout_context_pos_embed

    def _embed_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        if not self.trajectory_conditioned:
            raise RuntimeError("trajectory conditioning was not enabled for this predictor")
        if trajectories.ndim not in (3, 4) or trajectories.shape[-1] != 3:
            raise ValueError(
                "trajectories must have shape [B,T,3] or [B,P,T,3], "
                f"got {tuple(trajectories.shape)}"
            )
        return self.trajectory_embed(trajectories.flatten(start_dim=-2))

    def forward(
        self, current_latent: torch.Tensor, trajectory: torch.Tensor | None = None
    ):
        """
        Args:
            current_latent: [B, C, H, W] current vit feature map.
            trajectory: optional observed GT trajectory [B,T,3]. Only this
                branch receives a full future-latent reconstruction target.
        Returns:
            list of [B, C, H, W], one per future offset.
        """
        batch_size = current_latent.shape[0]
        feat_h, feat_w = self.spatial_hw
        context = self._context_tokens(current_latent)

        trajectory_token = None
        if trajectory is not None:
            if trajectory.ndim != 3:
                raise ValueError(
                    "full latent prediction accepts one trajectory per sample "
                    f"[B,T,3], got {tuple(trajectory.shape)}"
                )
            trajectory = trajectory.to(
                device=current_latent.device, dtype=current_latent.dtype
            )
            trajectory_token = self._embed_trajectories(trajectory)[:, None, :]

        predicted_maps = []
        for offset_idx in range(self.num_offsets):
            future_query = self.future_queries[offset_idx].expand(batch_size, -1, -1)
            future_query = future_query + self.future_pos_embed
            if trajectory_token is not None:
                future_query = future_query + trajectory_token
            sequence = torch.cat([context, future_query], dim=1)
            sequence = self.blocks(sequence)
            future_tokens = sequence[:, self.num_patches :, :]
            future_tokens = self.predictor_proj(self.predictor_norm(future_tokens))
            predicted_maps.append(
                tokens_to_spatial_map(future_tokens, feat_h, feat_w)
            )
        return predicted_maps

    def predict_rollout_features(
        self, current_latent: torch.Tensor, trajectories: torch.Tensor
    ) -> torch.Tensor:
        """Return [B,P,D] proposal features supervised only by scorer losses."""
        if trajectories.ndim != 4:
            raise ValueError(
                "proposal rollout expects trajectories [B,P,T,3], "
                f"got {tuple(trajectories.shape)}"
            )
        context = self._rollout_context_tokens(current_latent)
        trajectories = trajectories.to(
            device=current_latent.device, dtype=current_latent.dtype
        )
        proposal_queries = self.rollout_trajectory_embed(
            trajectories.flatten(start_dim=-2)
        )
        rollout = self.rollout_blocks(proposal_queries, context)
        return self.rollout_proj(self.rollout_norm(rollout))
