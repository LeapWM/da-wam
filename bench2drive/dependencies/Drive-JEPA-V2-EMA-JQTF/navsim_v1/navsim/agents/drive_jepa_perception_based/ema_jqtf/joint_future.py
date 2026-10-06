"""Joint-query trajectory-conditioned future latent predictor."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _spatial_map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    return feature_map.flatten(2).transpose(1, 2)


def _tokens_to_spatial_map(
    tokens: torch.Tensor, spatial_hw: Tuple[int, int]
) -> torch.Tensor:
    height, width = spatial_hw
    batch_size, token_count, channels = tokens.shape
    if token_count != height * width:
        raise ValueError(
            f"expected {height * width} spatial tokens, got {token_count}"
        )
    return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


class JointQueryFutureHead(nn.Module):
    """Predict candidate future tokens and one supervised full future latent.

    Expensive current-scene encoding is shared.  Each proposal runs only a
    short ``T x HW`` cross-attention path.  A full ``C x H x W`` latent is
    decoded for the matched proposal during training, or for the selected
    proposal during inference.
    """

    def __init__(
        self,
        token_dim: int,
        query_dim: int,
        future_dim: int,
        spatial_hw: Tuple[int, int] = (16, 32),
        num_poses: int = 8,
        candidate_layers: int = 2,
        full_decoder_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if future_dim % num_heads != 0:
            raise ValueError("future_dim must be divisible by num_heads")
        if candidate_layers <= 0 or full_decoder_layers <= 0:
            raise ValueError("future decoder depths must be positive")

        self.token_dim = int(token_dim)
        self.query_dim = int(query_dim)
        self.future_dim = int(future_dim)
        self.spatial_hw = tuple(int(value) for value in spatial_hw)
        self.num_patches = self.spatial_hw[0] * self.spatial_hw[1]
        self.num_poses = int(num_poses)

        self.context_proj = nn.Linear(token_dim, future_dim)
        self.context_norm = nn.LayerNorm(future_dim)
        self.context_pos = nn.Parameter(
            torch.zeros(1, self.num_patches, future_dim)
        )

        self.query_norm = nn.LayerNorm(query_dim)
        self.query_proj = nn.Linear(query_dim, future_dim)
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, 1, self.num_poses, future_dim)
        )

        candidate_layer = nn.TransformerDecoderLayer(
            d_model=future_dim,
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
        self.candidate_norm = nn.LayerNorm(future_dim)

        self.full_future_queries = nn.Parameter(
            torch.zeros(1, self.num_patches, future_dim)
        )
        full_layer = nn.TransformerDecoderLayer(
            d_model=future_dim,
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
        self.full_norm = nn.LayerNorm(future_dim)
        self.full_proj = nn.Linear(future_dim, token_dim)

        nn.init.trunc_normal_(self.context_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.full_future_queries, std=0.02)

    def encode_context(self, current_latent: torch.Tensor) -> torch.Tensor:
        """Return shared projected current-scene tokens ``[B,HW,D]``."""
        if current_latent.ndim != 4:
            raise ValueError(
                "current_latent must be [B,C,H,W], got "
                f"{tuple(current_latent.shape)}"
            )
        if tuple(current_latent.shape[-2:]) != self.spatial_hw:
            raise ValueError(
                f"expected spatial size {self.spatial_hw}, got "
                f"{tuple(current_latent.shape[-2:])}"
            )
        if current_latent.shape[1] != self.token_dim:
            raise ValueError(
                f"expected {self.token_dim} latent channels, got "
                f"{current_latent.shape[1]}"
            )
        context = self.context_proj(_spatial_map_to_tokens(current_latent))
        return self.context_norm(context + self.context_pos)

    def predict_candidate_tokens(
        self,
        current_latent: torch.Tensor,
        joint_queries: torch.Tensor,
    ) -> torch.Tensor:
        """Return proposal-conditioned compact future tokens ``[B,P,T,D]``."""
        if joint_queries.ndim != 4:
            raise ValueError(
                "joint_queries must be [B,P,T,D], got "
                f"{tuple(joint_queries.shape)}"
            )
        batch_size, proposal_count, time_count, query_dim = joint_queries.shape
        if time_count != self.num_poses or query_dim != self.query_dim:
            raise ValueError(
                f"expected joint query suffix ({self.num_poses},{self.query_dim}), "
                f"got ({time_count},{query_dim})"
            )
        if current_latent.shape[0] != batch_size:
            raise ValueError("current latent and joint query batch sizes differ")

        context = self.encode_context(current_latent)
        queries = self.query_proj(self.query_norm(joint_queries))
        queries = queries + self.temporal_pos
        queries = queries.reshape(
            batch_size * proposal_count, time_count, self.future_dim
        )
        memory = (
            context[:, None]
            .expand(-1, proposal_count, -1, -1)
            .reshape(
                batch_size * proposal_count,
                self.num_patches,
                self.future_dim,
            )
        )
        future_tokens = self.candidate_decoder(queries, memory)
        future_tokens = self.candidate_norm(future_tokens)
        return future_tokens.reshape(
            batch_size,
            proposal_count,
            time_count,
            self.future_dim,
        )

    @staticmethod
    def match_proposals(
        proposals: torch.Tensor, target_trajectory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Min-of-P L1 matching used by the existing trajectory loss."""
        if proposals.ndim != 4 or target_trajectory.ndim != 3:
            raise ValueError("proposals/target must be [B,P,T,3]/[B,T,3]")
        distances = torch.linalg.vector_norm(
            proposals - target_trajectory[:, None], ord=1, dim=-1
        ).mean(dim=-1)
        matched_distance, matched_index = distances.min(dim=1)
        return matched_index.detach(), matched_distance

    @staticmethod
    def _gather_candidates(
        candidate_tokens: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        batch_size = candidate_tokens.shape[0]
        if indices.shape != (batch_size,):
            raise ValueError(
                f"indices must be [{batch_size}], got {tuple(indices.shape)}"
            )
        batch_index = torch.arange(batch_size, device=candidate_tokens.device)
        return candidate_tokens[batch_index, indices]

    def decode_full_latent(
        self,
        current_latent: torch.Tensor,
        selected_candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one candidate per sample into ``[B,C,H,W]``."""
        if selected_candidate_tokens.ndim != 3:
            raise ValueError(
                "selected candidate tokens must be [B,T,D], got "
                f"{tuple(selected_candidate_tokens.shape)}"
            )
        context = self.encode_context(current_latent)
        future_queries = context + self.full_future_queries
        decoded = self.full_decoder(future_queries, selected_candidate_tokens)
        decoded = self.full_proj(self.full_norm(decoded))
        return _tokens_to_spatial_map(decoded, self.spatial_hw)

    def decode_matched_latent(
        self,
        current_latent: torch.Tensor,
        candidate_tokens: torch.Tensor,
        proposals: torch.Tensor,
        target_trajectory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matched_index, matched_distance = self.match_proposals(
            proposals, target_trajectory
        )
        matched_tokens = self._gather_candidates(candidate_tokens, matched_index)
        predicted_latent = self.decode_full_latent(
            current_latent, matched_tokens
        )
        return predicted_latent, matched_index, matched_distance

    def decode_selected_latent(
        self,
        current_latent: torch.Tensor,
        candidate_tokens: torch.Tensor,
        selected_index: torch.Tensor,
    ) -> torch.Tensor:
        selected_tokens = self._gather_candidates(
            candidate_tokens, selected_index
        )
        return self.decode_full_latent(current_latent, selected_tokens)
