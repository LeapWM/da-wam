from typing import Dict
import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from .score_module.scorer import Scorer
from .traj_refiner import Traj_refiner
from .bevformer.simple_image_encoder import ImgEncoder
from .bevformer.pretrain_vit_encoder import FrozenPretrainVitEncoder
from .bevformer.future_vit_token_predictor import FutureVitTokenPredictor
from .bevformer.transformer_decoder import MLP
from .drive_jepa_config import DriveJEPAConfig
from .ema_jqtf import EMAJQTFCandidateScorer, JointQueryFutureHead
from .ema_jqtf.formula_selection import (
    FORMULA_SELECTION_MODES,
    formula_progress_selection,
)
from .ema_jqtf.target_pair import FUTURE_TARGET_PAIR_MODES
from navsim.agents.drive_jepa_perception_free.vjepa_encoder import get_embed_dim


class FutureLatentPredictor(nn.Module):
    """Conv-based predictor for future latents in projected_map or pretrain vit space."""

    def __init__(self, channels: int = 256, num_offsets: int = 2, bottleneck: int = 0):
        super().__init__()
        self.num_offsets = num_offsets
        if bottleneck > 0 and bottleneck < channels:
            hidden = bottleneck
            self.predictor = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels * num_offsets, kernel_size=3, padding=1),
            )
        else:
            self.predictor = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels * num_offsets, kernel_size=3, padding=1),
            )

    def forward(self, current_latent: torch.Tensor):
        """
        Args:
            current_latent: [B, C, H, W]
        Returns:
            list of [B, C, H, W] predicted latents, one per future offset
        """
        out = self.predictor(current_latent)  # [B, C*num_offsets, H, W]
        return out.chunk(self.num_offsets, dim=1)


class DriveJEPAModel(nn.Module):
    def __init__(self, config: DriveJEPAConfig):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3

        self._backbone = ImgEncoder(config)

        self.command_num=config.command_num

        # pose(3) + velocity(2) + acceleration(2) + command one-hot.
        # NAVSIM keeps command_num=4 (11 total); Bench2Drive uses its six
        # official RoadOption classes (13 total).
        self.hist_encoding = nn.Linear(7 + config.command_num, config.tf_d_model)

        self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)

        ref_num=config.ref_num

        shared_refiner=Traj_refiner(config)

        self._trajectory_head=nn.ModuleList([shared_refiner for _ in range(ref_num) ] )

        self._ema_jqtf_enabled = config.future_integration == "ema_jqtf"
        if config.future_integration not in ("none", "scorer_pool", "ema_jqtf"):
            raise ValueError(
                f"Unsupported future_integration={config.future_integration!r}; "
                "expected 'none', 'scorer_pool', or 'ema_jqtf'"
            )
        if getattr(config, "future_diag_mode", "none") not in ("none", "off", "shuffle", "persistence"):
            raise ValueError(
                f"Unsupported future_diag_mode={config.future_diag_mode!r}; "
                "expected 'none' | 'off' | 'shuffle' | 'persistence'."
            )
        if config.future_integration != "none" and not config.predict_future:
            raise ValueError("future_integration requires predict_future=True")
        if config.future_target_mode not in ("frozen_pretrain_vit", "ema_online_encoder"):
            raise ValueError(
                f"Unsupported future_target_mode={config.future_target_mode!r}; "
                "expected 'frozen_pretrain_vit' or 'ema_online_encoder'."
            )
        if config.future_target_pair_mode not in FUTURE_TARGET_PAIR_MODES:
            raise ValueError(
                f"Unsupported future_target_pair_mode="
                f"{config.future_target_pair_mode!r}; expected one of "
                f"{FUTURE_TARGET_PAIR_MODES}"
            )
        if (
            config.future_target_pair_mode == "current_future"
            and tuple(int(value) for value in config.future_frame_offsets) != (1,)
        ):
            raise ValueError(
                "future_target_pair_mode='current_future' requires "
                "future_frame_offsets=(1,)"
            )
        if self._ema_jqtf_enabled:
            head_mode = config.ema_jqtf_scorer_head_mode
            if head_mode not in (
                "base_delta",
                "direct_final",
                "formula_progress",
            ):
                raise ValueError(
                    "ema_jqtf_scorer_head_mode must be 'base_delta', "
                    "'direct_final', or 'formula_progress', got "
                    f"{head_mode!r}"
                )
            if (
                getattr(config, "scorer_formula_counterfactual", False)
                and head_mode != "formula_progress"
            ):
                raise ValueError(
                    "scorer_formula_counterfactual requires the "
                    "formula_progress scorer head"
                )
            if head_mode == "direct_final":
                if float(config.sub_score_weight) != 0.0:
                    raise ValueError(
                        "EMA-JQTF direct_final requires sub_score_weight=0"
                    )
                if (
                    getattr(config, "scorer_anchor_auxiliary", False)
                    and float(config.scorer_anchor_subscore_weight) != 0.0
                ):
                    raise ValueError(
                        "EMA-JQTF direct_final anchors support final-score "
                        "supervision only; set scorer_anchor_subscore_weight=0"
                    )
            if head_mode == "formula_progress":
                if float(config.sub_score_weight) <= 0.0:
                    raise ValueError(
                        "EMA-JQTF formula_progress requires "
                        "sub_score_weight>0"
                    )
                if float(config.final_score_weight) != 0.0:
                    raise ValueError(
                        "EMA-JQTF formula_progress has no Direct-Final head; "
                        "set final_score_weight=0"
                    )
                if not config.scorer_pdm_formula_inference:
                    raise ValueError(
                        "EMA-JQTF formula_progress requires "
                        "scorer_pdm_formula_inference=True"
                    )
                formula_counterfactual = getattr(
                    config, "scorer_formula_counterfactual", False
                )
                if (
                    getattr(config, "scorer_anchor_auxiliary", False)
                    and not formula_counterfactual
                ):
                    raise ValueError(
                        "formula_progress anchors require the isolated "
                        "scorer_formula_counterfactual recipe"
                    )
                if formula_counterfactual and (
                    not getattr(config, "scorer_anchor_auxiliary", False)
                    or config.scorer_anchor_sampling
                    != "formula_counterfactual"
                ):
                    raise ValueError(
                        "scorer_formula_counterfactual requires "
                        "scorer_anchor_auxiliary=True and "
                        "scorer_anchor_sampling='formula_counterfactual'"
                    )
                if formula_counterfactual and float(
                    config.scorer_ranking_weight
                ) != 0.0:
                    raise ValueError(
                        "Formula counterfactual V1 is pointwise-only; set "
                        "scorer_ranking_weight=0"
                    )
                if float(config.scorer_progress_scale) <= 0.0:
                    raise ValueError(
                        "scorer_progress_scale must be positive"
                    )
                if config.scorer_formula_progress_mode not in (
                    "symmetric",
                    "official_threshold",
                    "one_sided_no5m",
                ):
                    raise ValueError(
                        "scorer_formula_progress_mode must be 'symmetric', "
                        "'official_threshold', or 'one_sided_no5m'"
                    )
                if config.scorer_formula_metric_loss_type not in (
                    "smooth_l1",
                    "bce",
                    "focal",
                ):
                    raise ValueError(
                        "scorer_formula_metric_loss_type must be "
                        "'smooth_l1', 'bce', or 'focal'"
                    )
                if float(config.scorer_formula_focal_gamma) < 0.0:
                    raise ValueError(
                        "scorer_formula_focal_gamma must be non-negative"
                    )
                if not 0.0 < float(config.scorer_formula_focal_alpha) < 1.0:
                    raise ValueError(
                        "scorer_formula_focal_alpha must be within (0,1)"
                    )
                if (
                    config.scorer_formula_selection_mode
                    not in FORMULA_SELECTION_MODES
                ):
                    raise ValueError(
                        "scorer_formula_selection_mode must be one of "
                        f"{FORMULA_SELECTION_MODES}, got "
                        f"{config.scorer_formula_selection_mode!r}"
                    )
                if int(config.scorer_formula_safety_topk) <= 0:
                    raise ValueError(
                        "scorer_formula_safety_topk must be positive"
                    )
                if int(config.ema_jqtf_pdm_query_layers) <= 0:
                    raise ValueError(
                        "ema_jqtf_pdm_query_layers must be positive"
                    )
                if float(
                    config.ema_jqtf_formula_scorer_lr_scale
                ) <= 0.0:
                    raise ValueError(
                        "ema_jqtf_formula_scorer_lr_scale must be positive"
                    )
                if float(
                    config.ema_jqtf_formula_future_lr_scale
                ) <= 0.0:
                    raise ValueError(
                        "ema_jqtf_formula_future_lr_scale must be positive"
                    )
                if int(
                    config.ema_jqtf_formula_lr_milestone
                ) <= 0:
                    raise ValueError(
                        "ema_jqtf_formula_lr_milestone must be positive"
                    )
                if not 0.0 < float(
                    config.ema_jqtf_formula_lr_gamma
                ) < 1.0:
                    raise ValueError(
                        "ema_jqtf_formula_lr_gamma must be within (0,1)"
                    )
            gradient_mode = config.ema_jqtf_scorer_gradient_mode
            if gradient_mode not in ("detached", "representation", "full"):
                raise ValueError(
                    "ema_jqtf_scorer_gradient_mode must be 'detached', "
                    f"'representation', or 'full', got {gradient_mode!r}"
                )
            if not config.predict_future:
                raise ValueError("ema_jqtf requires predict_future=True")
            if config.future_predictor_arch != "ema_jqtf":
                raise ValueError(
                    "ema_jqtf integration requires "
                    "future_predictor_arch='ema_jqtf'"
                )
            if config.future_target_mode != "ema_online_encoder":
                raise ValueError(
                    "ema_jqtf requires future_target_mode='ema_online_encoder'"
                )
            if config.future_target_pair_mode != "current_future":
                raise ValueError(
                    "ema_jqtf requires future_target_pair_mode='current_future'"
                )
            if config.future_latent_space != "lora_to_pretrain_vit":
                raise ValueError(
                    "ema_jqtf currently requires "
                    "future_latent_space='lora_to_pretrain_vit'"
                )
            if config.ref_num < 1:
                raise ValueError("ema_jqtf requires at least one trajectory refiner")
            if not config.future_gt_trajectory_conditioned:
                raise ValueError(
                    "ema_jqtf requires future_gt_trajectory_conditioned=True "
                    "to activate matched-query supervision"
                )
            if not config.scorer_traj_conditioned:
                raise ValueError(
                    "ema_jqtf requires scorer_traj_conditioned=True"
                )
            if (
                config.scorer_pdm_formula_inference
                and head_mode != "formula_progress"
            ):
                raise ValueError(
                    "EMA-JQTF base_delta/direct_final deploy a learned final "
                    "head; set scorer_pdm_formula_inference=False"
                )
            if not config.run_future_at_inference:
                raise ValueError(
                    "ema_jqtf scorer requires candidate future tokens during "
                    "deployment; set run_future_at_inference=True"
                )
        if getattr(config, "future_gt_trajectory_conditioned", False):
            if not config.predict_future:
                raise ValueError("future_gt_trajectory_conditioned requires predict_future=True")
            if config.future_predictor_arch not in ("jepa_token", "ema_jqtf"):
                raise ValueError(
                    "future_gt_trajectory_conditioned requires "
                    "future_predictor_arch='jepa_token' or 'ema_jqtf'"
                )
            if config.future_integration not in ("scorer_pool", "ema_jqtf"):
                raise ValueError(
                    "future_gt_trajectory_conditioned requires "
                    "future_integration='scorer_pool' or 'ema_jqtf'"
                )
            if not getattr(config, "scorer_traj_conditioned", False):
                raise ValueError(
                    "future_gt_trajectory_conditioned requires scorer_traj_conditioned=True"
                )
        if getattr(config, "scorer_anchor_auxiliary", False):
            if not getattr(config, "future_gt_trajectory_conditioned", False):
                raise ValueError(
                    "scorer_anchor_auxiliary requires future_gt_trajectory_conditioned=True"
                )
            if not getattr(config, "scorer_traj_conditioned", False):
                raise ValueError(
                    "scorer_anchor_auxiliary requires scorer_traj_conditioned=True"
                )
            if float(getattr(config, "scorer_ranking_weight", 0)) != 0.0:
                raise ValueError(
                    "hard-anchor experiments require scorer_ranking_weight=0"
                )
            if config.scorer_anchor_num_local_hard < 0 or config.scorer_anchor_num_balanced <= 0:
                raise ValueError("anchor sample counts must be non-negative with balanced > 0")
            if not 0 <= config.scorer_anchor_num_gt_local_hard <= config.scorer_anchor_num_local_hard:
                raise ValueError(
                    "scorer_anchor_num_gt_local_hard must be within [0, num_local_hard]"
                )
            if (
                config.scorer_anchor_num_gt_local_hard > 0
                and config.scorer_anchor_sampling != "hard_balanced"
            ):
                raise ValueError("GT-local-hard requires hard_balanced sampling")
            if config.scorer_anchor_nearest_per_gt <= 0:
                raise ValueError("scorer_anchor_nearest_per_gt must be positive")
            if config.scorer_anchor_loss_weight < 0:
                raise ValueError("scorer_anchor_loss_weight must be non-negative")
            if config.scorer_anchor_subscore_weight < 0:
                raise ValueError(
                    "scorer_anchor_subscore_weight must be non-negative"
                )
            if config.scorer_anchor_final_weight < 0:
                raise ValueError(
                    "scorer_anchor_final_weight must be non-negative"
                )

        if config.future_target_mode == "ema_online_encoder":
            if not config.predict_future:
                raise ValueError("future_target_mode='ema_online_encoder' requires predict_future=True")
            if config.future_latent_space != "lora_to_pretrain_vit":
                raise ValueError(
                    "future_target_mode='ema_online_encoder' is currently supported only "
                    "with future_latent_space='lora_to_pretrain_vit'."
                )
            self._ema_target_backbone = copy.deepcopy(self._backbone)
            self._freeze_future_target_ema()

        # Future latent prediction
        future_channels = None
        if config.predict_future:
            num_offsets = len(config.future_frame_offsets)
            if config.future_latent_space in ("pretrain_vit", "lora_to_pretrain_vit"):
                future_channels = get_embed_dim(config.vjepa_version, config.image_architecture)
                if (
                    config.future_latent_space == "pretrain_vit"
                    or config.future_target_mode == "frozen_pretrain_vit"
                ):
                    self._frozen_pretrain_vit = FrozenPretrainVitEncoder(config)
                if config.future_predictor_arch == "ema_jqtf":
                    if num_offsets != 1:
                        raise ValueError(
                            "ema_jqtf currently supports one future offset"
                        )
                    self._future_predictor = JointQueryFutureHead(
                        token_dim=future_channels,
                        query_dim=config.tf_d_model,
                        future_dim=config.future_predictor_dim,
                        spatial_hw=(16, 32),
                        num_poses=config.num_poses,
                        candidate_layers=config.ema_jqtf_candidate_layers,
                        full_decoder_layers=(
                            config.ema_jqtf_full_decoder_layers
                        ),
                        num_heads=config.future_predictor_heads,
                        ffn_dim=config.tf_d_ffn,
                        dropout=config.tf_dropout,
                    )
                elif config.future_predictor_arch == "jepa_token":
                    self._future_predictor = FutureVitTokenPredictor(
                        token_dim=future_channels,
                        num_offsets=num_offsets,
                        predictor_dim=config.future_predictor_dim,
                        depth=config.future_predictor_depth,
                        num_heads=config.future_predictor_heads,
                        trajectory_conditioned=getattr(
                            config, "future_gt_trajectory_conditioned", False
                        ),
                        trajectory_dim=config.num_poses * self.state_size,
                        rollout_dim=config.tf_d_model,
                        rollout_depth=getattr(config, "future_rollout_layers", 2),
                    )
                elif config.future_predictor_arch == "conv":
                    self._future_predictor = FutureLatentPredictor(
                        channels=future_channels,
                        num_offsets=num_offsets,
                        bottleneck=config.future_latent_bottleneck,
                    )
                else:
                    raise ValueError(
                        f"Unsupported future_predictor_arch={config.future_predictor_arch!r}. "
                        "Expected 'conv', 'jepa_token', or 'ema_jqtf'."
                    )
            elif config.future_latent_space == "projected_map":
                self._future_predictor = FutureLatentPredictor(
                    channels=config.tf_d_model,
                    num_offsets=num_offsets,
                    bottleneck=0,
                )
            else:
                raise ValueError(
                    f"Unsupported future_latent_space={config.future_latent_space!r}. "
                    "Expected 'projected_map' or 'pretrain_vit'."
                )

        if self._ema_jqtf_enabled:
            self.scorer = EMAJQTFCandidateScorer(
                query_dim=config.tf_d_model,
                future_dim=config.future_predictor_dim,
                scene_dim=config.tf_d_model,
                future_context_dim=int(future_channels),
                num_poses=config.num_poses,
                scene_hw=(16, 32),
                num_heads=config.tf_num_head,
                ffn_dim=config.tf_d_ffn,
                scene_layers=config.ema_jqtf_scorer_layers,
                external_layers=config.ema_jqtf_external_adapter_layers,
                dropout=config.tf_dropout,
                alpha_init=config.ema_jqtf_scorer_alpha_init,
                head_mode=config.ema_jqtf_scorer_head_mode,
                pdm_query_layers=config.ema_jqtf_pdm_query_layers,
            )
        else:
            self.scorer = Scorer(config)

        self.b2d=config.b2d
        self.transform = self.make_transform()
        if getattr(config, "scorer_only_training", False):
            self._freeze_for_scorer_only_training()

    def _freeze_for_scorer_only_training(self) -> None:
        """Freeze generator/backbone/full-latent head; train scorer + rollout only."""
        if not getattr(self._config, "scorer_traj_conditioned", False):
            raise ValueError("scorer_only_training requires scorer_traj_conditioned=True")
        if not getattr(self._config, "future_gt_trajectory_conditioned", False):
            raise ValueError(
                "scorer_only_training requires future_gt_trajectory_conditioned=True"
            )
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.scorer.parameters():
            parameter.requires_grad = True
        for name, parameter in self._future_predictor.named_parameters():
            if name.startswith("rollout_"):
                parameter.requires_grad = True

    def make_transform(self):
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([normalize])

    @staticmethod
    def _camera_to_float(camera: torch.Tensor) -> torch.Tensor:
        """Accept compact uint8 B2D caches and legacy float NAVSIM caches."""
        if camera.dtype == torch.uint8:
            return camera.float().div_(255.0)
        if not camera.is_floating_point():
            return camera.float()
        return camera

    def has_future_target_ema(self) -> bool:
        return hasattr(self, "_ema_target_backbone")

    def _freeze_future_target_ema(self) -> None:
        for param in self._ema_target_backbone.parameters():
            param.requires_grad = False
        self._ema_target_backbone.eval()

    @torch.no_grad()
    def update_future_target_ema(self) -> None:
        if not self.has_future_target_ema():
            return
        decay = float(self._config.future_ema_decay)
        target_params = dict(self._ema_target_backbone.named_parameters())
        online_params = dict(self._backbone.named_parameters())
        for name, target_param in target_params.items():
            online_param = online_params[name]
            target_param.mul_(decay).add_(online_param, alpha=1.0 - decay)
        target_buffers = dict(self._ema_target_backbone.named_buffers())
        online_buffers = dict(self._backbone.named_buffers())
        for name, target_buffer in target_buffers.items():
            if name in online_buffers:
                target_buffer.copy_(online_buffers[name])
        self._ema_target_backbone.eval()

    @torch.no_grad()
    def _encode_future_target_ema(self, camera_input: torch.Tensor) -> torch.Tensor:
        self._ema_target_backbone.eval()
        ema_out = self._ema_target_backbone(camera_input, img_metas={})
        return ema_out[5]

    def encode_future_target(self, camera_input: torch.Tensor) -> torch.Tensor:
        """Encode future frames into the configured future latent space."""
        if (
            self._config.future_latent_space == "lora_to_pretrain_vit"
            and self._config.future_target_mode == "ema_online_encoder"
        ):
            return self._encode_future_target_ema(camera_input)
        if self._config.future_latent_space in ("pretrain_vit", "lora_to_pretrain_vit"):
            return self._frozen_pretrain_vit(camera_input)
        fut_out = self._backbone(camera_input, img_metas={})
        return fut_out[-1]

    def _future_context_latent(self, camera_feature, projected_map, lora_vit_map):
        if self._config.future_latent_space == "lora_to_pretrain_vit":
            return lora_vit_map
        if self._config.future_latent_space == "pretrain_vit":
            with torch.no_grad():
                return self._frozen_pretrain_vit(camera_feature)
        return projected_map

    def _predict_future_latents(self, camera_feature, projected_map, lora_vit_map):
        context = self._future_context_latent(camera_feature, projected_map, lora_vit_map)
        return self._future_predictor(context)

    def predict_gt_future_latents(
        self, current_latent: torch.Tensor, gt_trajectory: torch.Tensor
    ):
        """Predict the full latent supervised only by the observed GT trajectory."""
        if self._ema_jqtf_enabled:
            raise RuntimeError(
                "EMA-JQTF must use predict_matched_future_latents so the same "
                "Q3 winner produces trajectory and future latent"
            )
        if not getattr(self._config, "future_gt_trajectory_conditioned", False):
            raise RuntimeError("GT trajectory conditioning is not enabled")
        return self._future_predictor(current_latent, trajectory=gt_trajectory)

    def predict_matched_future_latents(
        self,
        current_latent: torch.Tensor,
        candidate_future_tokens: torch.Tensor,
        proposals: torch.Tensor,
        gt_trajectory: torch.Tensor,
    ):
        """Decode the min-of-P matched Q3 future latent.

        The discrete argmin is detached inside ``JointQueryFutureHead`` while
        the selected future loss remains differentiable with respect to the
        matched Q3, shared BEV refiners, FutureHead, and online backbone.
        """
        if not self._ema_jqtf_enabled:
            raise RuntimeError("matched-query prediction requires EMA-JQTF")
        predicted, matched_index, matched_distance = (
            self._future_predictor.decode_matched_latent(
                current_latent,
                candidate_future_tokens,
                proposals,
                gt_trajectory,
            )
        )
        return [predicted], matched_index, matched_distance

    def predict_proposal_rollout_features(
        self, current_latent: torch.Tensor, proposals: torch.Tensor
    ) -> torch.Tensor:
        """Compact proposal features; scorer losses are their only supervision."""
        if self._ema_jqtf_enabled:
            raise RuntimeError(
                "EMA-JQTF proposal features are produced from Q3 in forward"
            )
        if not getattr(self._config, "future_gt_trajectory_conditioned", False):
            raise RuntimeError("proposal rollout conditioning is not enabled")
        return self._future_predictor.predict_rollout_features(current_latent, proposals)

    def score_external_trajectories(
        self,
        trajectories: torch.Tensor,
        nearest_proposal_pose_features: torch.Tensor,
        projected_map: torch.Tensor,
        future_context_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Training-only score path for sampled anchors; returns [B,A,6]."""
        if not getattr(self._config, "scorer_anchor_auxiliary", False):
            raise RuntimeError("scorer anchor auxiliary path is disabled")
        if not self.training:
            raise RuntimeError("external anchor scoring is training-only")
        if self._ema_jqtf_enabled:
            # The nearest generated slot is intentionally ignored.  The
            # scorer-owned adapter builds a packet from the true anchor and
            # detached current scene/future context.
            return self.scorer.score_external_trajectories(
                trajectories,
                projected_map.detach(),
                future_context_latent.detach(),
            )
        anchor_rollout_features = self.predict_proposal_rollout_features(
            future_context_latent, trajectories
        )
        return self.scorer.score_external_trajectories(
            trajectories,
            nearest_proposal_pose_features,
            projected_map,
            anchor_rollout_features,
        )

    def _apply_future_diag(self, predicted_future, projected_map, lora_vit_map):
        """Inference-time counterfactual on the future latent fed to the Scorer.

        Only active during eval (self.training is False); training always uses the
        raw predicted latent so the diagnostic never perturbs the learned weights.
        """
        diag = getattr(self._config, "future_diag_mode", "none")
        if diag == "none" or self.training:
            return predicted_future
        if diag == "off":
            return None
        if diag == "shuffle":
            batch_size = predicted_future.shape[0]
            if batch_size <= 1:
                return predicted_future
            perm = torch.randperm(batch_size, device=predicted_future.device)
            # avoid identity permutation for tiny batches
            if bool(torch.all(perm == torch.arange(batch_size, device=perm.device))):
                perm = torch.roll(perm, shifts=1, dims=0)
            return predicted_future[perm]
        if diag == "persistence":
            if self._config.future_latent_space == "projected_map":
                return projected_map
            return lora_vit_map
        return predicted_future

    def _apply_proposal_rollout_diag(self, rollout_features):
        """Eval-only intervention for proposal-conditioned rollout features."""
        diag = getattr(self._config, "future_diag_mode", "none")
        if diag == "none" or self.training:
            return rollout_features
        if diag == "off":
            return None
        if diag == "shuffle":
            batch_size = rollout_features.shape[0]
            if batch_size <= 1:
                return rollout_features
            perm = torch.randperm(batch_size, device=rollout_features.device)
            if bool(torch.all(perm == torch.arange(batch_size, device=perm.device))):
                perm = torch.roll(perm, shifts=1, dims=0)
            return rollout_features[perm]
        # The rollout branch is already conditioned on current context, so
        # persistence is identical to its normal input definition.
        return rollout_features

    def _should_run_future_predictor(self) -> bool:
        if not (self._config.predict_future and hasattr(self, "_future_predictor")):
            return False
        if self.training:
            return True
        return self._config.run_future_at_inference

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features['lidar2img'] = features['lidar2img'][:, 1:2]
        ego_status: torch.Tensor = features["ego_status"][:,-1]
        
        cam_f_2 = self._camera_to_float(features['camera_feature_2'])
        cam_f_1 = self._camera_to_float(features['camera_feature_1'])
        cam_f_2 = self.transform(cam_f_2)
        cam_f_1 = self.transform(cam_f_1)
        camera_feature = torch.cat([cam_f_2[:, None], cam_f_1[:, None]], dim=1)

        batch_size = ego_status.shape[0]

        if self.b2d:
            ego_status[:,1:3]=0

        image_feature_with_map = self._backbone(camera_feature,img_metas=features)  # 6-tuple
        projected_map = image_feature_with_map[4]  # [B, 256, 16, 32]
        lora_vit_map = image_feature_with_map[5]  # [B, 1024, 16, 32], grad to LoRA
        image_feature = image_feature_with_map[:4]   # 4-tuple, backward compatible for refiner

        output={}
        if self._ema_jqtf_enabled:
            # The target cache supplies t+0.5.  Retaining normalized t lets the
            # EMA teacher receive the required [t,t+0.5] clip without a cache
            # format change.  Validation loss needs the same contract; NavTest
            # does not contain future targets and therefore never runs teacher.
            output["future_target_current_frame"] = cam_f_1.detach()

        predicted_futures = None
        future_context_latent = None
        if (
            self._config.future_integration in ("scorer_pool", "ema_jqtf")
            and self._should_run_future_predictor()
        ):
            if getattr(self._config, "future_gt_trajectory_conditioned", False):
                future_context_latent = self._future_context_latent(
                    camera_feature, projected_map, lora_vit_map
                )
                output["future_context_latent"] = future_context_latent
            else:
                predicted_futures = self._predict_future_latents(
                    camera_feature, projected_map, lora_vit_map
                )
                output["predicted_future_latents"] = predicted_futures

        ego_feature=self.hist_encoding(ego_status)[:,None]

        bev_feature =ego_feature+self.init_feature.weight[None]

        proposal_list = []
        joint_query_feature = None

        for i, refine in enumerate(self._trajectory_head):
            if self._ema_jqtf_enabled:
                # On the final iteration this is Q3: the exact pre-decode query
                # that produces proposal^3.  The returned feature is Q4.
                joint_query_feature = bev_feature
            bev_feature, proposal_list = refine(bev_feature, proposal_list,image_feature)

        proposals=proposal_list[-1]

        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        candidate_future_tokens = None
        if self._ema_jqtf_enabled:
            if future_context_latent is None or joint_query_feature is None:
                raise RuntimeError(
                    "EMA-JQTF requires current latent and final pre-decode Q3"
                )
            joint_queries = joint_query_feature.reshape(
                batch_size,
                self._config.proposal_num,
                self.poses_num,
                self._config.tf_d_model,
            )
            candidate_future_tokens = (
                self._future_predictor.predict_candidate_tokens(
                    future_context_latent, joint_queries
                )
            )
            output["joint_query_features"] = joint_queries
            output["candidate_future_tokens"] = candidate_future_tokens
            output["projected_map"] = projected_map

        if self.training and getattr(self._config, "scorer_anchor_auxiliary", False):
            # Training-only retained state. Keep the full pose dimension for
            # nearest-slot gather now and future current-encoder fusion later.
            output["proposal_pose_features"] = bev_feature.reshape(
                batch_size,
                self._config.proposal_num,
                self.poses_num,
                self._config.tf_d_model,
            )
            output["projected_map"] = projected_map

        future_latent_for_scorer = None
        proposal_future_features = None
        if self._config.future_integration == "scorer_pool":
            if predicted_futures is not None:
                future_latent_for_scorer = self._apply_future_diag(
                    predicted_futures[0], projected_map, lora_vit_map
                )
            elif future_context_latent is not None:
                proposal_future_features = self.predict_proposal_rollout_features(
                    future_context_latent, proposals
                )
                proposal_future_features = self._apply_proposal_rollout_diag(
                    proposal_future_features
                )
                if proposal_future_features is not None:
                    output["proposal_future_features"] = proposal_future_features

        if self._ema_jqtf_enabled:
            gradient_mode = self._config.ema_jqtf_scorer_gradient_mode
            if gradient_mode == "detached":
                scorer_trajectories = proposals.detach()
                scorer_pose_features = bev_feature.detach()
                scorer_scene = projected_map.detach()
                scorer_future = candidate_future_tokens.detach()
                scorer_context = future_context_latent.detach()
            elif gradient_mode == "representation":
                # The label is evaluated on the current proposal coordinates.
                # Do not use that fixed-label gradient to directly move the
                # coordinates, but let score supervision shape the generated
                # query, scene representation, and future representation.
                scorer_trajectories = proposals.detach()
                scorer_pose_features = bev_feature
                scorer_scene = projected_map
                scorer_future = candidate_future_tokens
                scorer_context = future_context_latent
            elif gradient_mode == "full":
                scorer_trajectories = proposals
                scorer_pose_features = bev_feature
                scorer_scene = projected_map
                scorer_future = candidate_future_tokens
                scorer_context = future_context_latent
            else:
                raise RuntimeError(
                    "invalid EMA-JQTF scorer gradient mode at forward: "
                    f"{gradient_mode!r}"
                )
            (
                pred_logit,
                pred_logit2,
                pred_agents_states,
                pred_area_logit,
                bev_semantic_map,
                agent_states,
                agent_labels,
                pred_pdm_progress_norm,
            ) = self.scorer(
                scorer_trajectories,
                scorer_pose_features,
                scorer_scene,
                scorer_future,
                scorer_context,
            )
        else:
            pred_pdm_progress_norm = None
            (
                pred_logit,
                pred_logit2,
                pred_agents_states,
                pred_area_logit,
                bev_semantic_map,
                agent_states,
                agent_labels,
            ) = self.scorer(
                proposals,
                bev_feature,
                future_latent_for_scorer,
                projected_map,
                proposal_future_features,
            )

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels
        output["pred_pdm_progress_norm"] = pred_pdm_progress_norm

        selection_score = None
        if (
            self._ema_jqtf_enabled
            and self._config.ema_jqtf_scorer_head_mode
            == "formula_progress"
        ):
            if pred_pdm_progress_norm is None:
                raise RuntimeError(
                    "formula_progress scorer did not return PDM progress"
                )
            metric_scores = torch.sigmoid(pred_logit[..., :4])
            noc = metric_scores[..., 0]
            dac = metric_scores[..., 1]
            ttc = metric_scores[..., 2]
            comfort = metric_scores[..., 3]
            raw_progress_norm = pred_logit[..., 4]
            formula_selection = formula_progress_selection(
                metric_scores,
                raw_progress_norm,
                pred_pdm_progress_norm,
                progress_scale=float(self._config.scorer_progress_scale),
                mode=self._config.scorer_formula_selection_mode,
                safety_topk=int(
                    self._config.scorer_formula_safety_topk
                ),
                progress_mode=self._config.scorer_formula_progress_mode,
                progress_distance_threshold=float(
                    self._config.scorer_progress_distance_threshold
                ),
                safe_progress_tie_break_weight=float(
                    self._config.scorer_formula_safe_progress_tie_break_weight
                ),
            )
            ep = formula_selection.ego_progress
            pdm_score = formula_selection.formula_score
            selection_score = formula_selection.selection_score
            progress_scale = float(
                self._config.scorer_progress_scale
            )
            output["pred_metric_scores"] = metric_scores
            output["pred_raw_progress_norm"] = raw_progress_norm
            output["pred_raw_progress"] = (
                raw_progress_norm * progress_scale
            )
            output["pred_pdm_progress"] = (
                pred_pdm_progress_norm * progress_scale
            )
            output["pred_ep"] = ep
            output["pred_formula_safety_rank"] = (
                formula_selection.safety_rank
            )
        elif self._config.scorer_traj_conditioned and self._config.scorer_pdm_formula_inference:
            # PDM-like reward from factorized subscore heads [NOC,DAC,EP,TTC,comfort]
            s = torch.sigmoid(pred_logit)
            noc, dac, ep, ttc, comfort = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4]
            pdm_score = noc * dac * (5.0 * ttc + 5.0 * ep + 2.0 * comfort) / 12.0
        elif pred_logit2 is not None:
            pdm_score=(torch.sigmoid(pred_logit)+torch.sigmoid(pred_logit2))[:,:,-1]/2
        else:
            pdm_score=torch.sigmoid(pred_logit)[:,:,-1]

        if selection_score is None:
            selection_score = pdm_score
        token = torch.argmax(selection_score, dim=1)
        batch_index = torch.arange(batch_size, device=proposals.device)
        trajectory = proposals[batch_index, token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score
        output["selection_score"] = selection_score
        output["selected_index"] = token

        if (
            self._ema_jqtf_enabled
            and not self.training
            and self._config.ema_jqtf_decode_selected_future
        ):
            output["selected_future_latent"] = (
                self._future_predictor.decode_selected_latent(
                    future_context_latent,
                    candidate_future_tokens,
                    token,
                )
            )

        if (
            predicted_futures is None
            and future_context_latent is None
            and self._should_run_future_predictor()
        ):
            predicted_futures = self._predict_future_latents(
                camera_feature, projected_map, lora_vit_map
            )
            output["predicted_future_latents"] = predicted_futures

        return output

    def train(self, mode: bool = True):
        super().train(mode)
        if self.has_future_target_ema():
            self._ema_target_backbone.eval()
        return self
