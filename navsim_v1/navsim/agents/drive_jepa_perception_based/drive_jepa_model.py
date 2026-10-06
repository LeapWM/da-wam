import copy
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from .score_module.scorer import Scorer
from .traj_refiner import Traj_refiner
from .bevformer.simple_image_encoder import ImgEncoder
from .bevformer.transformer_decoder import MLP
from .drive_jepa_config import DriveJEPAConfig
from .posttraj_future import PostTrajectoryFutureHead, PostTrajectoryScorer


class DriveJEPAModel(nn.Module):
    def __init__(self, config: DriveJEPAConfig):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3

        if config.posttraj_future_enabled:
            offsets = tuple(
                int(value)
                for value in config.posttraj_future_frame_offsets
            )
            if offsets not in ((1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)):
                raise ValueError(
                    "post-trajectory horizons must be contiguous offsets 1..H, H<=4"
                )
            if config.posttraj_scorer_temporal_layers <= 0:
                raise ValueError(
                    "posttraj_scorer_temporal_layers must be positive"
                )
            if (
                config.posttraj_full_candidate_latent
                and offsets != (1,)
            ):
                raise ValueError(
                    "full-candidate latent mode currently requires the "
                    "single [t,t+0.5] target"
                )

        self._backbone = ImgEncoder(config)

        if config.posttraj_future_enabled:
            self._ema_target_backbone = copy.deepcopy(self._backbone)
            self._freeze_future_target_ema()

        self.command_num=config.command_num

        self.hist_encoding = nn.Linear(7 + config.command_num if config.b2d else 11, config.tf_d_model)

        self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)

        ref_num=config.ref_num

        shared_refiner=Traj_refiner(config)

        self._trajectory_head=nn.ModuleList([shared_refiner for _ in range(ref_num) ] )

        if config.posttraj_future_enabled:
            self._future_predictor = PostTrajectoryFutureHead(
                dim=config.tf_d_model,
                num_poses=config.num_poses,
                spatial_hw=(16, 32),
                num_heads=config.tf_num_head,
                ffn_dim=config.tf_d_ffn,
                candidate_layers=config.posttraj_future_candidate_layers,
                full_decoder_layers=(
                    config.posttraj_future_full_decoder_layers
                ),
                decode_all_candidates=(
                    config.posttraj_full_candidate_latent
                ),
                dropout=config.tf_dropout,
            )
            self.scorer = PostTrajectoryScorer(config)
        else:
            self.scorer = Scorer(config)

        self.b2d=config.b2d
        self.transform = self.make_transform()

    def make_transform(self):
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([normalize])

    @staticmethod
    def _image_feature_to_map(image_feature: tuple) -> torch.Tensor:
        """Recover ``[B,D,16,32]`` from PB-G's BEVFormer tuple."""
        flat_feature = image_feature[0]
        if flat_feature.ndim != 4 or flat_feature.shape[0] != 1:
            raise ValueError(
                "PB-G image feature must be [1,HW,B,D], got "
                f"{tuple(flat_feature.shape)}"
            )
        tokens = flat_feature[0]  # [HW,B,D]
        if tokens.shape[0] != 16 * 32:
            raise ValueError(
                f"expected 512 image tokens, got {tokens.shape[0]}"
            )
        return tokens.permute(1, 2, 0).reshape(
            tokens.shape[1], tokens.shape[2], 16, 32
        )

    def has_future_target_ema(self) -> bool:
        return hasattr(self, "_ema_target_backbone")

    def _freeze_future_target_ema(self) -> None:
        for parameter in self._ema_target_backbone.parameters():
            parameter.requires_grad = False
        self._ema_target_backbone.eval()

    @torch.no_grad()
    def update_future_target_ema(self) -> None:
        if not self.has_future_target_ema():
            return
        decay = float(self._config.posttraj_future_ema_decay)
        online_parameters = dict(self._backbone.named_parameters())
        for name, target_parameter in self._ema_target_backbone.named_parameters():
            target_parameter.mul_(decay).add_(
                online_parameters[name], alpha=1.0 - decay
            )
        online_buffers = dict(self._backbone.named_buffers())
        for name, target_buffer in self._ema_target_backbone.named_buffers():
            if name in online_buffers:
                target_buffer.copy_(online_buffers[name])
        self._ema_target_backbone.eval()

    @torch.no_grad()
    def encode_future_target(self, camera_pair: torch.Tensor) -> torch.Tensor:
        if not self.has_future_target_ema():
            raise RuntimeError("post-trajectory EMA target is disabled")
        self._ema_target_backbone.eval()
        target_feature = self._ema_target_backbone(
            camera_pair, img_metas={}
        )
        return self._image_feature_to_map(target_feature)

    def predict_matched_future_latent(
        self,
        scene_map: torch.Tensor,
        candidate_future_tokens: torch.Tensor,
        proposals: torch.Tensor,
        target_trajectory: torch.Tensor,
        horizon_offset: int = 1,
    ):
        return self._future_predictor.decode_matched_latent(
            scene_map,
            candidate_future_tokens,
            proposals,
            target_trajectory,
            horizon_offset=horizon_offset,
        )

    def score_external_trajectories(
        self,
        trajectories: torch.Tensor,
        scene_map: torch.Tensor,
    ) -> torch.Tensor:
        """Counterfactual-ready scorer path with no generator gradients."""
        if not self._config.posttraj_future_enabled:
            raise RuntimeError("post-trajectory future path is disabled")
        trajectories = trajectories.detach()
        scene_map = scene_map.detach()
        pose_tokens = self.scorer.external_pose_tokens(
            trajectories, scene_map
        )
        future_tokens = self._future_predictor.predict_candidate_tokens(
            scene_map, trajectories, pose_tokens
        )
        full_latents = None
        if self._config.posttraj_full_candidate_latent:
            full_latents = self._future_predictor.decode_all_latents(
                scene_map, future_tokens
            )
        return self.scorer.score_external(
            trajectories, scene_map, future_tokens, full_latents
        )

    def configure_posttraj_stage2_trainable(self) -> None:
        """Freeze Stage A and open only scorer/candidate-future parameters."""
        if not self._config.posttraj_future_enabled:
            raise RuntimeError("PostTraj Stage 2 requires the future path")
        for parameter in self.parameters():
            parameter.requires_grad = False

        # The final scorer is the deployment head. Agent/area auxiliaries are
        # intentionally excluded from Stage 2 supervision and optimization.
        for name, parameter in self.scorer.named_parameters():
            if not name.startswith(("pred_col_agent.", "pred_area.")):
                parameter.requires_grad = True

        candidate_prefixes = (
            "scene_norm.",
            "scene_pos",
            "native_query_norm.",
            "trajectory_embed.",
            "query_fuse.",
            "temporal_pos",
            "candidate_decoder.",
            "candidate_norm.",
        )
        for name, parameter in self._future_predictor.named_parameters():
            if name.startswith(candidate_prefixes):
                parameter.requires_grad = True

        trainable = [name for name, p in self.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("PostTraj Stage 2 has no trainable parameters")
        if any(
            name.startswith(
                ("_backbone.", "_ema_target_backbone.", "_trajectory_head.")
            )
            for name in trainable
        ):
            raise AssertionError("Stage 2 opened a frozen generator parameter")
        if any("_future_predictor.full_" in name for name in trainable):
            raise AssertionError("Stage 2 opened the full latent decoder")

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features['lidar2img'] = features['lidar2img'][:, 1:2]
        ego_status: torch.Tensor = features["ego_status"][:,-1]
        
        cam_f_2 = features['camera_feature_2']
        cam_f_1 = features['camera_feature_1']
        cam_f_2 = self.transform(cam_f_2)
        cam_f_1 = self.transform(cam_f_1)
        camera_feature = torch.cat([cam_f_2[:, None], cam_f_1[:, None]], dim=1)

        batch_size = ego_status.shape[0]

        if self.b2d and not self._config.b2d_expert_selection:
            ego_status = ego_status.clone()
            ego_status[:,1:3]=0

        image_feature = self._backbone(camera_feature,img_metas=features)  # b,64,64,64
        scene_map = None
        if self._config.posttraj_future_enabled:
            scene_map = self._image_feature_to_map(image_feature)

        output={}

        ego_feature=self.hist_encoding(ego_status)[:,None]

        bev_feature =ego_feature+self.init_feature.weight[None]

        proposal_list = []

        for i, refine in enumerate(self._trajectory_head):
            bev_feature, proposal_list = refine(bev_feature, proposal_list,image_feature)

        proposals=proposal_list[-1]

        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        if self._config.posttraj_future_enabled:
            proposal_inputs = (
                proposals.detach()
                if self._config.posttraj_detach_trajectory_inputs
                else proposals
            )
            pose_tokens = bev_feature.reshape(
                batch_size,
                self._config.proposal_num,
                self._config.num_poses,
                self._config.tf_d_model,
            )
            candidate_future_tokens = (
                self._future_predictor.predict_candidate_tokens(
                    scene_map,
                    proposal_inputs,
                    pose_tokens,
                )
            )
            output["candidate_future_tokens"] = candidate_future_tokens
            candidate_full_latents = None
            if self._config.posttraj_full_candidate_latent:
                candidate_full_latents = (
                    self._future_predictor.decode_all_latents(
                        scene_map, candidate_future_tokens
                    )
                )
                output["candidate_full_latents"] = candidate_full_latents
            output["future_scene_map"] = scene_map
            output["future_target_current_frame"] = cam_f_1.detach()
            scorer_output = self.scorer(
                proposal_inputs,
                bev_feature,
                scene_map,
                candidate_future_tokens,
                candidate_full_latents,
            )
        else:
            scorer_output = self.scorer(proposals, bev_feature)

        (
            pred_logit,
            pred_logit2,
            pred_agents_states,
            pred_area_logit,
            bev_semantic_map,
            agent_states,
            agent_labels,
        ) = scorer_output

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels

        if pred_logit2 is not None:
            pdm_score=(torch.sigmoid(pred_logit)+torch.sigmoid(pred_logit2))[:,:,-1]/2
        else:
            pdm_score=torch.sigmoid(pred_logit)[:,:,-1]

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output

    def train(self, mode: bool = True):
        super().train(mode)
        if self.has_future_target_ema():
            self._ema_target_backbone.eval()
        if mode and self._config.posttraj_stage2_enabled:
            self._backbone.eval()
            self._trajectory_head.eval()
            self._future_predictor.full_decoder.eval()
            self._future_predictor.full_norm.eval()
        return self
