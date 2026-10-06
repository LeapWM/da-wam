"""Trajectory-first future latent prediction and candidate scoring modules."""

from typing import Tuple

import torch
import torch.nn as nn

from ..bevformer.transformer_decoder import MLP


class PostTrajectoryFutureHead(nn.Module):
    """Predict proposal-specific future tokens after trajectories exist.

    Current-scene encoding is shared.  The 32 proposal paths are folded into
    the batch dimension for a single batched Transformer call; no V-JEPA
    backbone is run independently per proposal.
    """

    def __init__(
        self,
        dim: int = 256,
        num_poses: int = 8,
        spatial_hw: Tuple[int, int] = (16, 32),
        num_heads: int = 8,
        ffn_dim: int = 1024,
        candidate_layers: int = 2,
        full_decoder_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("future dim must be divisible by num_heads")
        if candidate_layers <= 0 or full_decoder_layers <= 0:
            raise ValueError("future decoder depths must be positive")

        self.dim = int(dim)
        self.num_poses = int(num_poses)
        self.spatial_hw = tuple(int(v) for v in spatial_hw)
        self.num_scene_tokens = self.spatial_hw[0] * self.spatial_hw[1]

        self.scene_norm = nn.LayerNorm(dim)
        self.scene_pos = nn.Parameter(
            torch.zeros(1, self.num_scene_tokens, dim)
        )
        self.native_query_norm = nn.LayerNorm(dim)
        self.trajectory_embed = MLP(3, ffn_dim, dim)
        self.query_fuse = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, 1, self.num_poses, dim)
        )

        candidate_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_decoder = nn.TransformerDecoder(
            candidate_layer, num_layers=candidate_layers
        )
        self.candidate_norm = nn.LayerNorm(dim)

        self.full_queries = nn.Parameter(
            torch.zeros(1, self.num_scene_tokens, dim)
        )
        full_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.full_decoder = nn.TransformerDecoder(
            full_layer, num_layers=full_decoder_layers
        )
        self.full_norm = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.scene_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.full_queries, std=0.02)

    def _scene_tokens(self, scene_map: torch.Tensor) -> torch.Tensor:
        if scene_map.ndim != 4:
            raise ValueError("scene_map must be [B,D,H,W]")
        if scene_map.shape[1] != self.dim:
            raise ValueError(
                f"expected scene dim {self.dim}, got {scene_map.shape[1]}"
            )
        if tuple(scene_map.shape[-2:]) != self.spatial_hw:
            raise ValueError(
                f"expected scene size {self.spatial_hw}, got "
                f"{tuple(scene_map.shape[-2:])}"
            )
        tokens = scene_map.flatten(2).transpose(1, 2)
        return self.scene_norm(tokens + self.scene_pos)

    def predict_candidate_tokens(
        self,
        scene_map: torch.Tensor,
        trajectories: torch.Tensor,
        native_pose_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return candidate future tokens shaped ``[B,P,T,D]``."""
        if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
            raise ValueError("trajectories must be [B,P,T,3]")
        batch_size, proposal_count, time_count = trajectories.shape[:3]
        if time_count != self.num_poses:
            raise ValueError(
                f"expected {self.num_poses} poses, got {time_count}"
            )
        if native_pose_tokens.ndim == 3:
            native_pose_tokens = native_pose_tokens.reshape(
                batch_size, proposal_count, time_count, self.dim
            )
        expected = (batch_size, proposal_count, time_count, self.dim)
        if tuple(native_pose_tokens.shape) != expected:
            raise ValueError(
                f"native pose tokens must be {expected}, got "
                f"{tuple(native_pose_tokens.shape)}"
            )

        trajectory_tokens = self.trajectory_embed(
            trajectories.to(dtype=native_pose_tokens.dtype)
        )
        queries = self.query_fuse(
            torch.cat(
                [self.native_query_norm(native_pose_tokens), trajectory_tokens],
                dim=-1,
            )
        )
        queries = queries + self.temporal_pos
        queries = queries.reshape(
            batch_size * proposal_count, time_count, self.dim
        )

        scene_tokens = self._scene_tokens(scene_map)
        memory = (
            scene_tokens[:, None]
            .expand(-1, proposal_count, -1, -1)
            .reshape(
                batch_size * proposal_count,
                self.num_scene_tokens,
                self.dim,
            )
        )
        future_tokens = self.candidate_decoder(queries, memory)
        future_tokens = self.candidate_norm(future_tokens)
        return future_tokens.reshape(
            batch_size, proposal_count, time_count, self.dim
        )

    @staticmethod
    def match_proposals(
        trajectories: torch.Tensor,
        target_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if trajectories.ndim != 4 or target_trajectory.ndim != 3:
            raise ValueError("trajectory tensors must be [B,P,T,3]/[B,T,3]")
        distances = torch.linalg.vector_norm(
            trajectories - target_trajectory[:, None], ord=1, dim=-1
        ).mean(dim=-1)
        matched_distance, matched_index = distances.min(dim=1)
        return matched_index.detach(), matched_distance

    @staticmethod
    def _gather_candidates(
        candidate_tokens: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = candidate_tokens.shape[0]
        batch_index = torch.arange(batch_size, device=candidate_tokens.device)
        return candidate_tokens[batch_index, indices]

    def decode_full_latent(
        self,
        scene_map: torch.Tensor,
        selected_candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if selected_candidate_tokens.ndim != 3:
            raise ValueError("selected candidate tokens must be [B,T,D]")
        scene_tokens = self._scene_tokens(scene_map)
        queries = scene_tokens + self.full_queries
        decoded = self.full_decoder(queries, selected_candidate_tokens)
        decoded = self.full_norm(decoded)
        height, width = self.spatial_hw
        return decoded.transpose(1, 2).reshape(
            scene_map.shape[0], self.dim, height, width
        )

    def decode_matched_latent(
        self,
        scene_map: torch.Tensor,
        candidate_tokens: torch.Tensor,
        trajectories: torch.Tensor,
        target_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matched_index, matched_distance = self.match_proposals(
            trajectories, target_trajectory
        )
        selected_tokens = self._gather_candidates(
            candidate_tokens, matched_index
        )
        predicted_latent = self.decode_full_latent(
            scene_map, selected_tokens
        )
        return predicted_latent, matched_index, matched_distance


class PostTrajectoryScorer(nn.Module):
    """PB-G-compatible heads with explicit trajectory/future/scene fusion."""

    def __init__(self, config) -> None:
        super().__init__()
        dim = int(config.tf_d_model)
        ffn_dim = int(config.tf_d_ffn)
        num_heads = int(config.tf_num_head)
        self.b2d = bool(config.b2d)
        self.num_poses = int(config.num_poses)
        self.score_num = 6

        self.native_norm = nn.LayerNorm(dim)
        self.trajectory_embed = MLP(3, ffn_dim, dim)
        self.future_norm = nn.LayerNorm(dim)
        self.pose_fuse = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=float(config.tf_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=int(config.posttraj_scorer_temporal_layers),
        )
        self.scene_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=float(config.tf_dropout),
            batch_first=True,
        )
        self.score_norm = nn.LayerNorm(dim)
        self.pred_score = MLP(dim, ffn_dim, self.score_num)

        # This adapter supplies Q4-shaped tokens for future counterfactual
        # trajectories, which do not own native iterative-refiner queries.
        self.external_temporal_pos = nn.Parameter(
            torch.zeros(1, 1, self.num_poses, dim)
        )
        self.external_scene_proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.external_temporal_pos, std=0.02)

        self.agent_pred = bool(config.agent_pred)
        if self.agent_pred:
            agent_out = 2 * 6 * 9 if self.b2d else 2 * 40 * 9
            self.pred_col_agent = MLP(dim, ffn_dim, agent_out)

        self.area_pred = bool(config.area_pred)
        if self.area_pred:
            area_out = 2 if self.b2d else 5 * 2
            self.pred_area = MLP(dim, ffn_dim, area_out)

        self.bev_map = False
        self.bev_agent = False

    @staticmethod
    def _scene_tokens(scene_map: torch.Tensor) -> torch.Tensor:
        return scene_map.flatten(2).transpose(1, 2)

    def external_pose_tokens(
        self,
        trajectories: torch.Tensor,
        scene_map: torch.Tensor,
    ) -> torch.Tensor:
        """Build Q4-shaped tokens without a nearest-proposal lookup."""
        if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
            raise ValueError("external trajectories must be [B,K,T,3]")
        scene_global = self.external_scene_proj(
            scene_map.flatten(2).mean(dim=-1)
        )
        return (
            self.trajectory_embed(trajectories.to(dtype=scene_map.dtype))
            + scene_global[:, None, None]
            + self.external_temporal_pos
        )

    def _score_features(
        self,
        trajectories: torch.Tensor,
        pose_tokens: torch.Tensor,
        scene_map: torch.Tensor,
        candidate_future_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, proposal_count, time_count = trajectories.shape[:3]
        pose_tokens = pose_tokens.reshape(
            batch_size, proposal_count, time_count, -1
        )
        trajectory_tokens = self.trajectory_embed(
            trajectories.to(dtype=pose_tokens.dtype)
        )
        fused_pose = self.pose_fuse(
            torch.cat(
                [
                    self.native_norm(pose_tokens),
                    trajectory_tokens,
                    self.future_norm(candidate_future_tokens),
                ],
                dim=-1,
            )
        )
        fused_pose = self.temporal_encoder(
            fused_pose.reshape(
                batch_size * proposal_count, time_count, -1
            )
        )
        proposal_feature = fused_pose.amax(dim=1).reshape(
            batch_size, proposal_count, -1
        )

        query = proposal_feature.reshape(
            batch_size * proposal_count, 1, -1
        )
        scene_tokens = self._scene_tokens(scene_map)
        memory = (
            scene_tokens[:, None]
            .expand(-1, proposal_count, -1, -1)
            .reshape(batch_size * proposal_count, scene_tokens.shape[1], -1)
        )
        scene_feature, _ = self.scene_attention(
            query, memory, memory, need_weights=False
        )
        scene_feature = scene_feature.reshape(
            batch_size, proposal_count, -1
        )
        return self.score_norm(proposal_feature + scene_feature)

    def forward(
        self,
        proposals: torch.Tensor,
        bev_feature: torch.Tensor,
        scene_map: torch.Tensor,
        candidate_future_tokens: torch.Tensor,
    ):
        batch_size, proposal_count, time_count = proposals.shape[:3]
        pose_tokens = bev_feature.reshape(
            batch_size, proposal_count, time_count, -1
        )
        fused_feature = self._score_features(
            proposals, pose_tokens, scene_map, candidate_future_tokens
        )
        pred_logit = self.pred_score(fused_feature)

        pred_agents_states = None
        pred_area_logit = None
        if self.training:
            # Preserve PB-G auxiliary supervision on its native Q4 features.
            native_feature = pose_tokens.amax(dim=2)
            if self.area_pred:
                pred_area_logit = self.pred_area(bev_feature)
            if self.agent_pred:
                pred_agents_states = self.pred_col_agent(native_feature).reshape(
                    batch_size,
                    proposal_count,
                    time_count,
                    -1,
                    2,
                    9,
                )

        return (
            pred_logit,
            None,
            pred_agents_states,
            pred_area_logit,
            None,
            None,
            None,
        )

    def score_external(
        self,
        trajectories: torch.Tensor,
        scene_map: torch.Tensor,
        candidate_future_tokens: torch.Tensor,
    ) -> torch.Tensor:
        pose_tokens = self.external_pose_tokens(trajectories, scene_map)
        fused = self._score_features(
            trajectories, pose_tokens, scene_map, candidate_future_tokens
        )
        return self.pred_score(fused)
