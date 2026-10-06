"""Unified generated/counterfactual scorer for EMA-JQTF."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class EMAJQTFCandidateScorer(nn.Module):
    """Score any candidate packet with shared heads.

    Generated candidates provide native Q4 pose tokens.  Counterfactual
    trajectories obtain same-shaped pose/future tokens from a scorer-owned
    adapter that reads detached scene features.  No nearest generated proposal
    is required.
    """

    def __init__(
        self,
        query_dim: int = 256,
        future_dim: int = 256,
        scene_dim: int = 256,
        future_context_dim: int = 1024,
        num_poses: int = 8,
        scene_hw: Tuple[int, int] = (16, 32),
        num_heads: int = 8,
        ffn_dim: int = 1024,
        scene_layers: int = 2,
        external_layers: int = 1,
        dropout: float = 0.0,
        alpha_init: float = 0.0,
        head_mode: str = "base_delta",
        pdm_query_layers: int = 1,
    ) -> None:
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        if head_mode not in (
            "base_delta",
            "direct_final",
            "formula_progress",
        ):
            raise ValueError(
                "head_mode must be 'base_delta', 'direct_final', or "
                "'formula_progress', got "
                f"{head_mode!r}"
            )
        if pdm_query_layers <= 0:
            raise ValueError("pdm_query_layers must be positive")
        self.head_mode = head_mode
        self.score_num = {
            "base_delta": 6,
            "direct_final": 1,
            # [NOC logit, DAC logit, TTC logit, comfort logit,
            #  non-negative normalized raw progress]
            "formula_progress": 5,
        }[head_mode]
        self.query_dim = int(query_dim)
        self.future_dim = int(future_dim)
        self.scene_dim = int(scene_dim)
        self.future_context_dim = int(future_context_dim)
        self.num_poses = int(num_poses)
        self.scene_hw = tuple(int(value) for value in scene_hw)
        self.scene_num_tokens = self.scene_hw[0] * self.scene_hw[1]

        self.pose_norm = nn.LayerNorm(query_dim)
        self.trajectory_embed = _MLP(
            num_poses * 3, ffn_dim, query_dim
        )
        self.future_proj = (
            nn.Identity()
            if future_dim == query_dim
            else nn.Linear(future_dim, query_dim)
        )
        self.future_cross_attention = nn.MultiheadAttention(
            query_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.scene_proj = (
            nn.Identity()
            if scene_dim == query_dim
            else nn.Linear(scene_dim, query_dim)
        )
        self.scene_pos = nn.Parameter(
            torch.zeros(1, self.scene_num_tokens, query_dim)
        )
        scene_layer = nn.TransformerDecoderLayer(
            d_model=query_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scene_decoder = nn.TransformerDecoder(
            scene_layer, num_layers=scene_layers
        )
        self.score_norm = nn.LayerNorm(query_dim)

        if self.head_mode == "direct_final":
            # All trajectory/future/scene information participates from the
            # first optimization step.  The scorer input boundary remains
            # detached in DriveJEPAModel, so this does not reconnect scorer
            # gradients to the generator.
            self.final_head = _MLP(query_dim, ffn_dim, 1)
        elif self.head_mode == "formula_progress":
            # Four continuous PDM metric heads plus candidate raw progress.
            # These heads all read the proposal-conditioned fused feature.
            self.metric_heads = nn.ModuleList(
                _MLP(query_dim, ffn_dim, 1) for _ in range(4)
            )
            self.raw_progress_head = _MLP(query_dim, ffn_dim, 1)

            # PDM baseline progress is a property of the scene, not of a
            # proposal.  One learned query reads only candidate-independent
            # BEV/current-latent memory and produces one value per scene.
            self.pdm_query = nn.Parameter(
                torch.zeros(1, 1, query_dim)
            )
            self.pdm_context_proj = nn.Linear(
                future_context_dim, query_dim
            )
            self.pdm_context_pos = nn.Parameter(
                torch.zeros(1, self.scene_num_tokens, query_dim)
            )
            pdm_layer = nn.TransformerDecoderLayer(
                d_model=query_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.pdm_scene_decoder = nn.TransformerDecoder(
                pdm_layer, num_layers=pdm_query_layers
            )
            self.pdm_norm = nn.LayerNorm(query_dim)
            self.pdm_progress_head = _MLP(
                query_dim, ffn_dim, 1
            )
        else:
            # Checkpoint-compatible Stage-A path.
            self.base_heads = nn.ModuleList(
                _MLP(query_dim, ffn_dim, 1) for _ in range(self.score_num)
            )
            self.delta_heads = nn.ModuleList(
                _MLP(query_dim, ffn_dim, 1) for _ in range(self.score_num)
            )
            self.alpha_scorer = nn.Parameter(torch.tensor(float(alpha_init)))

        # Scorer-owned counterfactual adapter.  It sees only detached scene
        # tensors, so its loss cannot modify the generator/backbone.
        self.external_pose_embed = nn.Linear(3, query_dim)
        self.external_temporal_pos = nn.Parameter(
            torch.zeros(1, 1, num_poses, query_dim)
        )
        external_scene_layer = nn.TransformerDecoderLayer(
            d_model=query_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.external_scene_decoder = nn.TransformerDecoder(
            external_scene_layer, num_layers=external_layers
        )
        self.external_future_context_proj = nn.Linear(
            future_context_dim, query_dim
        )
        self.external_future_pos = nn.Parameter(
            torch.zeros(1, self.scene_num_tokens, query_dim)
        )
        external_future_layer = nn.TransformerDecoderLayer(
            d_model=query_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.external_future_decoder = nn.TransformerDecoder(
            external_future_layer, num_layers=external_layers
        )

        nn.init.trunc_normal_(self.scene_pos, std=0.02)
        nn.init.trunc_normal_(self.external_temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.external_future_pos, std=0.02)
        if self.head_mode == "formula_progress":
            nn.init.trunc_normal_(self.pdm_query, std=0.02)
            nn.init.trunc_normal_(self.pdm_context_pos, std=0.02)

    @staticmethod
    def _map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.flatten(2).transpose(1, 2)

    def _validate_scene_map(
        self, feature_map: torch.Tensor, expected_channels: int, name: str
    ) -> None:
        if feature_map.ndim != 4:
            raise ValueError(f"{name} must be [B,C,H,W]")
        if feature_map.shape[1] != expected_channels:
            raise ValueError(
                f"{name} expected {expected_channels} channels, got "
                f"{feature_map.shape[1]}"
            )
        if tuple(feature_map.shape[-2:]) != self.scene_hw:
            raise ValueError(
                f"{name} expected spatial size {self.scene_hw}, got "
                f"{tuple(feature_map.shape[-2:])}"
            )

    def _reshape_pose_tokens(
        self, pose_tokens: torch.Tensor, proposal_count: int
    ) -> torch.Tensor:
        if pose_tokens.ndim == 3:
            batch_size = pose_tokens.shape[0]
            expected = proposal_count * self.num_poses
            if pose_tokens.shape[1] != expected:
                raise ValueError(
                    f"expected {expected} flattened pose tokens, got "
                    f"{pose_tokens.shape[1]}"
                )
            pose_tokens = pose_tokens.reshape(
                batch_size, proposal_count, self.num_poses, self.query_dim
            )
        if pose_tokens.ndim != 4:
            raise ValueError("pose_tokens must be [B,P,T,D] or [B,P*T,D]")
        return pose_tokens

    def _score(
        self,
        trajectories: torch.Tensor,
        pose_tokens: torch.Tensor,
        projected_map: torch.Tensor,
        candidate_future_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
            raise ValueError("trajectories must be [B,K,T,3]")
        batch_size, candidate_count, time_count = trajectories.shape[:3]
        if time_count != self.num_poses:
            raise ValueError(
                f"expected {self.num_poses} trajectory poses, got {time_count}"
            )
        pose_tokens = self._reshape_pose_tokens(
            pose_tokens, candidate_count
        )
        expected_prefix = (
            batch_size,
            candidate_count,
            self.num_poses,
        )
        if tuple(pose_tokens.shape[:3]) != expected_prefix:
            raise ValueError("pose tokens do not align with trajectories")
        if tuple(candidate_future_tokens.shape[:3]) != expected_prefix:
            raise ValueError("future tokens do not align with trajectories")

        self._validate_scene_map(
            projected_map, self.scene_dim, "projected_map"
        )
        pose_tokens = self.pose_norm(pose_tokens)
        base_feature = pose_tokens.amax(dim=2)
        trajectory_feature = self.trajectory_embed(
            trajectories.reshape(batch_size, candidate_count, -1).to(
                dtype=base_feature.dtype
            )
        )
        score_query = base_feature + trajectory_feature

        flat_query = score_query.reshape(
            batch_size * candidate_count, 1, self.query_dim
        )
        flat_future = self.future_proj(candidate_future_tokens).reshape(
            batch_size * candidate_count,
            self.num_poses,
            self.query_dim,
        )
        future_feature, _ = self.future_cross_attention(
            flat_query, flat_future, flat_future, need_weights=False
        )
        future_feature = future_feature.reshape(
            batch_size, candidate_count, self.query_dim
        )

        scene_tokens = self.scene_proj(self._map_to_tokens(projected_map))
        scene_tokens = scene_tokens + self.scene_pos
        fused = self.scene_decoder(
            score_query + future_feature, scene_tokens
        )
        fused = self.score_norm(fused + base_feature)

        if self.head_mode == "direct_final":
            return self.final_head(fused)
        if self.head_mode == "formula_progress":
            metric_logits = torch.cat(
                [head(fused) for head in self.metric_heads], dim=-1
            )
            raw_progress = F.softplus(
                self.raw_progress_head(fused)
            )
            return torch.cat(
                [metric_logits, raw_progress], dim=-1
            )

        base_logits = torch.cat(
            [head(base_feature) for head in self.base_heads], dim=-1
        )
        delta_logits = torch.cat(
            [head(fused) for head in self.delta_heads], dim=-1
        )
        return base_logits + self.alpha_scorer * delta_logits

    def _predict_pdm_progress(
        self,
        projected_map: torch.Tensor,
        future_context_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one normalized PDM baseline progress per scene."""
        if self.head_mode != "formula_progress":
            raise RuntimeError(
                "PDM progress is available only in formula_progress mode"
            )
        self._validate_scene_map(
            projected_map, self.scene_dim, "projected_map"
        )
        self._validate_scene_map(
            future_context_latent,
            self.future_context_dim,
            "future_context_latent",
        )
        scene_tokens = self.scene_proj(
            self._map_to_tokens(projected_map)
        )
        scene_tokens = scene_tokens + self.scene_pos
        context_tokens = self.pdm_context_proj(
            self._map_to_tokens(future_context_latent)
        )
        context_tokens = context_tokens + self.pdm_context_pos
        memory = torch.cat(
            [scene_tokens, context_tokens], dim=1
        )
        query = self.pdm_query.expand(
            projected_map.shape[0], -1, -1
        )
        pdm_feature = self.pdm_scene_decoder(query, memory)
        pdm_feature = self.pdm_norm(pdm_feature)
        return F.softplus(
            self.pdm_progress_head(pdm_feature)
        ).squeeze(-1)

    def forward(
        self,
        trajectories: torch.Tensor,
        pose_tokens: torch.Tensor,
        projected_map: torch.Tensor,
        candidate_future_tokens: torch.Tensor,
        future_context_latent: Optional[torch.Tensor] = None,
    ):
        logits = self._score(
            trajectories,
            pose_tokens,
            projected_map,
            candidate_future_tokens,
        )
        pdm_progress = None
        if self.head_mode == "formula_progress":
            if future_context_latent is None:
                raise ValueError(
                    "formula_progress requires candidate-independent "
                    "future_context_latent"
                )
            pdm_progress = self._predict_pdm_progress(
                projected_map, future_context_latent
            )
        # Preserve the legacy scorer return contract.
        return logits, None, None, None, None, None, None, pdm_progress

    def _external_candidate_packet(
        self,
        trajectories: torch.Tensor,
        projected_map: torch.Tensor,
        future_context_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, candidate_count, time_count = trajectories.shape[:3]
        if time_count != self.num_poses:
            raise ValueError("external trajectories have wrong pose count")
        self._validate_scene_map(
            projected_map, self.scene_dim, "projected_map"
        )
        self._validate_scene_map(
            future_context_latent,
            self.future_context_dim,
            "future_context_latent",
        )

        scene_tokens = self.scene_proj(
            self._map_to_tokens(projected_map.detach())
        )
        scene_tokens = scene_tokens + self.scene_pos
        repeated_scene = (
            scene_tokens[:, None]
            .expand(-1, candidate_count, -1, -1)
            .reshape(
                batch_size * candidate_count,
                self.scene_num_tokens,
                self.query_dim,
            )
        )
        pose_tokens = self.external_pose_embed(
            trajectories.detach().to(dtype=scene_tokens.dtype)
        )
        pose_tokens = pose_tokens + self.external_temporal_pos
        flat_pose = pose_tokens.reshape(
            batch_size * candidate_count,
            self.num_poses,
            self.query_dim,
        )
        flat_pose = self.external_scene_decoder(flat_pose, repeated_scene)
        pose_tokens = flat_pose.reshape(
            batch_size,
            candidate_count,
            self.num_poses,
            self.query_dim,
        )

        future_scene = self.external_future_context_proj(
            self._map_to_tokens(future_context_latent.detach())
        )
        future_scene = future_scene + self.external_future_pos
        repeated_future = (
            future_scene[:, None]
            .expand(-1, candidate_count, -1, -1)
            .reshape(
                batch_size * candidate_count,
                self.scene_num_tokens,
                self.query_dim,
            )
        )
        external_future = self.external_future_decoder(
            flat_pose, repeated_future
        ).reshape(
            batch_size,
            candidate_count,
            self.num_poses,
            self.query_dim,
        )
        return pose_tokens, external_future

    def score_external_trajectories(
        self,
        trajectories: torch.Tensor,
        projected_map: torch.Tensor,
        future_context_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Score arbitrary trajectories with no nearest-proposal feature."""
        pose_tokens, future_tokens = self._external_candidate_packet(
            trajectories, projected_map, future_context_latent
        )
        return self._score(
            trajectories.detach(),
            pose_tokens,
            projected_map.detach(),
            future_tokens,
        )
