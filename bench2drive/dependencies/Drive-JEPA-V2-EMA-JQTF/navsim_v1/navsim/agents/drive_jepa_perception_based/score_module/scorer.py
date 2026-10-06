import torch
import torch.nn as nn
from einops import rearrange
from ..bevformer.bev_refiner import Bev_refiner
from ..bevformer.transformer_decoder import MyTransformeDecoder,MLP
from .map_head import MapHead
from navsim.agents.drive_jepa_perception_free.vjepa_encoder import get_embed_dim
import numpy as np

class Scorer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.b2d=config.b2d

        self.proposal_num=config.proposal_num
        self.score_num = 6
        self.tf_d_model = config.tf_d_model
        self.num_poses = config.num_poses
        self.state_size = 3
        self.future_integration = config.future_integration
        self.future_pool_fusion = config.future_pool_fusion
        self.future_scorer_detach = config.future_scorer_detach
        self.scorer_traj_conditioned = getattr(config, "scorer_traj_conditioned", False)
        self.future_gt_trajectory_conditioned = getattr(
            config, "future_gt_trajectory_conditioned", False
        )

        if self.future_integration not in ("none", "scorer_pool"):
            raise ValueError(f"Unsupported future_integration={self.future_integration!r}; only 'scorer_pool' is implemented")
        if self.future_integration == "scorer_pool":
            if self.future_pool_fusion not in ("add", "gate_add"):
                raise ValueError(f"Unsupported future_pool_fusion={self.future_pool_fusion!r}")
            if config.future_latent_space == "projected_map":
                future_channels = config.tf_d_model
            else:
                future_channels = get_embed_dim(config.vjepa_version, config.image_architecture)
            self.future_pool_proj = nn.Linear(future_channels, config.tf_d_model)
            if self.future_pool_fusion == "gate_add":
                self.future_pool_gate = nn.Parameter(torch.zeros(1))

        self.pred_score = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)
        self.double_score=config.double_score

        if self.double_score:
            self.pred_score2 = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)

        self.agent_pred= config.agent_pred

        if self.agent_pred:
            if self.b2d:
                self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn, 2*6* 9)
            else:
                self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn,2* 40 * 9)

        self.area_pred=config.area_pred

        if self.area_pred:
            if self.b2d:
                self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 2)
            else:
                self.pred_area =  MLP(config.tf_d_model, config.tf_d_ffn, 5*2)

        self.bev_map=config.bev_map
        self.bev_agent=config.bev_agent

        if config.bev_agent:
            self._agent_head=MyTransformeDecoder(config,config.num_bounding_boxes,6,trajenc=False)

        if config.bev_map:
            self.map_head=MapHead(config)

        # === SC-2: trajectory-conditioned scorer ===
        if self.scorer_traj_conditioned:
            d = config.tf_d_model
            # trajectory geometry encoder: each proposal [T,3] -> d
            self.traj_embed = nn.Sequential(
                nn.Linear(self.num_poses * self.state_size, config.tf_d_ffn),
                nn.ReLU(inplace=True),
                nn.Linear(config.tf_d_ffn, d),
            )
            feat_h, feat_w = 16, 32
            self.scene_num_tokens = feat_h * feat_w
            self.scene_pos_embed = nn.Parameter(torch.zeros(1, self.scene_num_tokens, d))
            self.scene_modality_embed = nn.Parameter(torch.zeros(2, d))  # 0=current, 1=future
            nn.init.trunc_normal_(self.scene_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.scene_modality_embed, std=0.02)
            self.use_future_scene = (
                getattr(config, "scorer_use_future_scene", True)
                and self.future_integration == "scorer_pool"
                and not self.future_gt_trajectory_conditioned
            )
            if self.future_gt_trajectory_conditioned:
                self.proposal_rollout_norm = nn.LayerNorm(d)
            if self.use_future_scene:
                if config.future_latent_space == "projected_map":
                    fut_ch = config.tf_d_model
                else:
                    fut_ch = get_embed_dim(config.vjepa_version, config.image_architecture)
                self.future_scene_proj = nn.Linear(fut_ch, d)
            dec_layer = nn.TransformerDecoderLayer(
                d_model=d,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
            )
            self.traj_xattn = nn.TransformerDecoder(dec_layer, config.scorer_xattn_layers)
            self.traj_residual = config.scorer_traj_residual
            # factorized PDM heads; order MUST match target_scores columns:
            # [NOC, DAC, EP, TTC, comfort, final]
            self._pdm_head_order = ["noc", "dac", "ep", "ttc", "comfort", "final"]
            self.pred_score_heads = nn.ModuleDict({
                name: MLP(d, config.tf_d_ffn, 1) for name in self._pdm_head_order
            })

    def _build_scene_memory(self, projected_map, future_latent):
        """[B,d,16,32] current (+ optional predicted future) -> [B,N,d] scene tokens."""
        cur = rearrange(projected_map, "b c h w -> b (h w) c")
        cur = cur + self.scene_pos_embed + self.scene_modality_embed[0][None, None]
        tokens = [cur]
        if self.use_future_scene and future_latent is not None:
            if self.future_scorer_detach:
                future_latent = future_latent.detach()
            fut = rearrange(future_latent, "b c h w -> b (h w) c")
            fut = self.future_scene_proj(fut)
            fut = fut + self.scene_pos_embed + self.scene_modality_embed[1][None, None]
            tokens.append(fut)
        return torch.cat(tokens, dim=1)

    def _forward_traj_conditioned(
        self,
        proposals,
        bev_feature,
        future_latent,
        projected_map,
        proposal_future_features=None,
    ):
        if projected_map is None:
            raise ValueError("scorer_traj_conditioned=True requires projected_map from the model forward")
        batch_size, p_size, t_size = proposals.shape[0], proposals.shape[1], proposals.shape[2]
        proposal_feature = bev_feature.reshape(batch_size, p_size, t_size, -1).amax(-2)  # [B,P,d]
        proposals = proposals.to(
            device=proposal_feature.device, dtype=proposal_feature.dtype
        )
        traj_geom = self.traj_embed(proposals.reshape(batch_size, p_size, -1))            # [B,P,d]
        query = proposal_feature + traj_geom
        if proposal_future_features is not None:
            if tuple(proposal_future_features.shape[:2]) != (batch_size, p_size):
                raise ValueError(
                    "proposal_future_features must align with [B,P], got "
                    f"{tuple(proposal_future_features.shape)} for {(batch_size, p_size)}"
                )
            if self.future_scorer_detach:
                proposal_future_features = proposal_future_features.detach()
            query = query + self.proposal_rollout_norm(proposal_future_features)
        scene_tokens = self._build_scene_memory(projected_map, future_latent)             # [B,N,d]
        tr_out = self.traj_xattn(query, scene_tokens)                                     # [B,P,d]
        if self.traj_residual:
            tr_out = tr_out + proposal_feature
        logits = [self.pred_score_heads[name](tr_out).squeeze(-1) for name in self._pdm_head_order]
        pred_logit = torch.stack(logits, dim=-1)  # [B,P,6] order [NOC,DAC,EP,TTC,comfort,final]

        pred_logit2 = pred_agents_states = pred_area_logit = bev_semantic_map = agent_states = agent_labels = None
        if self.training:
            if self.area_pred:
                pred_area_logit = self.pred_area(bev_feature)
            if self.agent_pred:
                pred_agents_states = self.pred_col_agent(proposal_feature).reshape(batch_size, p_size, t_size, -1, 2, 9)
            if self.bev_map:
                bev_semantic_map = self.map_head(bev_feature)
            if self.bev_agent:
                agents = self._agent_head(None, bev_feature)
                agent_states = agents[:, :, :-1]
                agent_labels = agents[:, :, -1]
        return pred_logit, pred_logit2, pred_agents_states, pred_area_logit, bev_semantic_map, agent_states, agent_labels

    def score_external_trajectories(
        self,
        trajectories,
        nearest_proposal_pose_features,
        projected_map,
        proposal_future_features,
    ):
        """Score training-only trajectories through the shared PDM heads.

        This bypasses agent/area/map auxiliary heads. The gathered generator
        slot is detached here, so anchor BCE cannot train the trajectory
        generator through its proposal features.
        """
        if not self.scorer_traj_conditioned:
            raise RuntimeError(
                "external trajectory scoring requires scorer_traj_conditioned=True"
            )
        if projected_map is None:
            raise ValueError("external trajectory scoring requires projected_map")
        if proposal_future_features is None:
            raise ValueError(
                "external trajectory scoring requires anchor rollout features"
            )
        if trajectories.ndim != 4 or trajectories.shape[-1] != self.state_size:
            raise ValueError(
                "trajectories must be [B,A,T,3], got "
                f"{tuple(trajectories.shape)}"
            )
        if nearest_proposal_pose_features.ndim != 4:
            raise ValueError(
                "nearest_proposal_pose_features must be [B,A,T,D], got "
                f"{tuple(nearest_proposal_pose_features.shape)}"
            )
        batch_size, anchor_count, time_count = trajectories.shape[:3]
        expected_prefix = (batch_size, anchor_count, time_count)
        if tuple(nearest_proposal_pose_features.shape[:3]) != expected_prefix:
            raise ValueError(
                "nearest proposal features do not align with external trajectories: "
                f"{tuple(nearest_proposal_pose_features.shape)} vs {expected_prefix}"
            )
        if tuple(proposal_future_features.shape[:2]) != (batch_size, anchor_count):
            raise ValueError(
                "proposal_future_features must align with [B,A], got "
                f"{tuple(proposal_future_features.shape)}"
            )

        pose_features = nearest_proposal_pose_features.detach()
        proposal_feature = pose_features.amax(dim=-2)
        trajectories = trajectories.to(
            device=proposal_feature.device, dtype=proposal_feature.dtype
        )
        trajectory_feature = self.traj_embed(
            trajectories.reshape(batch_size, anchor_count, -1)
        )
        query = (
            proposal_feature
            + trajectory_feature
            + self.proposal_rollout_norm(proposal_future_features)
        )
        scene_tokens = self._build_scene_memory(projected_map, future_latent=None)
        score_feature = self.traj_xattn(query, scene_tokens)
        if self.traj_residual:
            score_feature = score_feature + proposal_feature
        logits = [
            self.pred_score_heads[name](score_feature).squeeze(-1)
            for name in self._pdm_head_order
        ]
        return torch.stack(logits, dim=-1)

    def _fuse_future_pool(self, proposal_feature, future_latent):
        if self.future_integration != "scorer_pool" or future_latent is None:
            return proposal_feature
        if self.future_scorer_detach:
            future_latent = future_latent.detach()
        future_pool = future_latent.flatten(2).mean(-1)
        future_feature = self.future_pool_proj(future_pool)[:, None, :]
        if self.future_pool_fusion == "gate_add":
            future_feature = self.future_pool_gate * future_feature
        return proposal_feature + future_feature

    def forward(
        self,
        proposals,
        bev_feature,
        future_latent=None,
        projected_map=None,
        proposal_future_features=None,
    ):
        if self.scorer_traj_conditioned:
            return self._forward_traj_conditioned(
                proposals,
                bev_feature,
                future_latent,
                projected_map,
                proposal_future_features,
            )

        batch_size=len(proposals)
        p_size=proposals.shape[1]
        t_size=proposals.shape[2]

        proposal_feature = bev_feature.reshape(batch_size, p_size, t_size, -1).amax(-2)
        proposal_feature = self._fuse_future_pool(proposal_feature, future_latent)
        pred_logit = self.pred_score(proposal_feature).reshape(batch_size, -1, self.score_num)

        pred_logit2=pred_agents_states=pred_area_logit=bev_semantic_map=agent_states=agent_labels=None

        if self.double_score:
            pred_logit2 = self.pred_score2(proposal_feature).reshape(batch_size, -1, self.score_num)

        if  self.training:
            if self.area_pred:
                pred_area_logit = self.pred_area(bev_feature)

            if self.agent_pred:
                pred_agents_states = self.pred_col_agent(proposal_feature).reshape(batch_size,p_size,t_size,-1,2,9)

            if self.bev_map:
                bev_semantic_map = self.map_head(bev_feature)

            if self.bev_agent:
                agents =self._agent_head(None,bev_feature)
                agent_states = agents[:, :, :-1]
                agent_labels = agents[:, :, -1]

        return pred_logit,pred_logit2, pred_agents_states, pred_area_logit,bev_semantic_map,agent_states,agent_labels
