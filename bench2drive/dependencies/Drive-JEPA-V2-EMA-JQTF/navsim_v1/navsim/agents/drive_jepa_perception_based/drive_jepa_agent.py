from typing import Any, Dict, List, Optional, Union
import hashlib
import gzip

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import os
from pathlib import Path
import pickle
from navsim.agents.drive_jepa_perception_based.drive_jepa_model import DriveJEPAModel 
from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.agents.drive_jepa_perception_based.drive_jepa_features import (
    DriveJEPAFeatureBuilder,
    DriveJEPATargetBuilder,
    select_future_camera_features_for_training,
)
from navsim.agents.transfuser.transfuser_loss import _agent_loss
from .score_module.hard_anchor_sampler import (
    HardAnchorSampler,
    HardAnchorSamplerConfig,
    anchor_score_cache_path,
    load_anchor_scores,
)
from .score_module.formula_counterfactual_sampler import (
    FormulaCounterfactualSampler,
    FormulaCounterfactualSamplerConfig,
)
from .score_module.asymmetric_safety_loss import (
    AsymmetricSafetyLossOutput,
    asymmetric_safety_loss,
)
from .score_module.formula_metric_loss import (
    formula_metric_bce_per_head,
    formula_metric_focal_per_head,
)
from .ema_jqtf import build_future_target_camera_pair
from .ema_jqtf.formula_selection import formula_progress_selection
from .ema_jqtf.progress_ranking import (
    progress_ranking_loss,
    safe_raw_progress_tie_ranking_loss,
)
from .ema_jqtf.loss_schedule import bundled_loss_weight, scheduled_loss_weight


class FutureTargetEMACallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self._last_global_step = 0

    def on_train_start(self, trainer, pl_module) -> None:
        self._last_global_step = int(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        global_step = int(trainer.global_step)
        if global_step <= self._last_global_step:
            return
        agent = getattr(pl_module, "agent", None)
        if agent is not None and hasattr(agent, "update_future_target_ema"):
            agent.update_future_target_ema()
        self._last_global_step = global_step


class DriveJEPAAgent(AbstractAgent):
    def __init__(
            self,
            config: DriveJEPAConfig,
            lr: float,
            checkpoint_path: str = None,
            cache_data: bool = False,
    ):
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._cache_data = cache_data
        self._training_epoch = 0

        if not self._cache_data:
            self._pad_model = DriveJEPAModel(config)

        scorer_only_training = getattr(config, "scorer_only_training", False)
        joint_continuation_training = getattr(
            config, "joint_continuation_training", False
        )
        if scorer_only_training and joint_continuation_training:
            raise ValueError(
                "scorer_only_training and joint_continuation_training are "
                "mutually exclusive"
            )

        training_setup = (
            not self._cache_data
            and (
                self._checkpoint_path == ""
                or scorer_only_training
                or joint_continuation_training
            )
        )
        if training_setup:
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.b2d = config.b2d

            self.ray=True

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                if self.b2d:
                    self.worker = RayDistributedNoTorch(threads_per_node=8)
                else:
                    self.worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=16)
                self.worker_map=worker_map

            if self.b2d:
                from .score_module.compute_b2d_score import (
                    b2d_before_score,
                    get_scores,
                )

                def load_b2d_pickle(path_value, description):
                    if not path_value:
                        raise ValueError(
                            f"{description} is required when config.b2d=True"
                        )
                    path = Path(path_value).expanduser()
                    if not path.is_file():
                        raise FileNotFoundError(f"{description} not found: {path}")
                    opener = gzip.open if path.suffix == ".gz" else open
                    with opener(path, "rb") as file_obj:
                        return pickle.load(file_obj)

                self.train_metric_cache_paths = load_b2d_pickle(
                    config.b2d_train_scorer_cache_path,
                    "b2d_train_scorer_cache_path",
                )
                val_cache_path = (
                    config.b2d_val_scorer_cache_path
                    or config.b2d_train_scorer_cache_path
                )
                self.test_metric_cache_paths = load_b2d_pickle(
                    val_cache_path,
                    "b2d_val_scorer_cache_path",
                )
                self.b2d_map_infos = load_b2d_pickle(
                    config.b2d_map_cache_path,
                    "b2d_map_cache_path",
                )
                self.b2d_before_score = b2d_before_score
                self.get_scores = get_scores
            else:
                from .score_module.compute_navsim_score import get_scores

                metric_cache = MetricCacheLoader(
                    Path(
                        os.getenv("NAVSIM_EXP_ROOT")
                        + "/Drive-JEPA-cache/train_metric_cache"
                    )
                )
                self.train_metric_cache_paths = metric_cache.metric_cache_paths
                self.test_metric_cache_paths = metric_cache.metric_cache_paths
                self.get_scores = get_scores
        
        if not self._cache_data:
            poses = np.load(config.anchor_trajectory_path)
            self.anchors = poses[:, 4::5]
            if getattr(config, "scorer_anchor_auxiliary", False):
                if not hasattr(self, "train_metric_cache_paths"):
                    raise RuntimeError(
                        "anchor auxiliary training requires training metric-cache paths"
                    )
                sampling = config.scorer_anchor_sampling
                if sampling not in (
                    "hard_balanced",
                    "balanced",
                    "formula_counterfactual",
                ):
                    raise ValueError(
                        "scorer_anchor_sampling must be hard_balanced, balanced, "
                        "or formula_counterfactual, "
                        f"got {sampling!r}"
                    )
                if sampling == "formula_counterfactual":
                    if not getattr(
                        config, "scorer_formula_counterfactual", False
                    ):
                        raise ValueError(
                            "formula_counterfactual sampling requires "
                            "scorer_formula_counterfactual=True"
                        )
                    self._formula_counterfactual_sampler = (
                        FormulaCounterfactualSampler(
                            self.anchors,
                            FormulaCounterfactualSamplerConfig(
                                num_ttc_only=(
                                    config.scorer_formula_cf_num_ttc_only
                                ),
                                num_joint_unsafe=(
                                    config.scorer_formula_cf_num_joint_unsafe
                                ),
                                num_safe_rescue=(
                                    config.scorer_formula_cf_num_safe_rescue
                                ),
                                num_safe_progress=(
                                    config.scorer_formula_cf_num_safe_progress
                                ),
                                num_balanced=(
                                    config.scorer_formula_cf_num_balanced
                                ),
                                safe_metric_min=(
                                    config.scorer_formula_cf_safe_metric_min
                                ),
                                failure_metric_max=(
                                    config.scorer_formula_cf_failure_metric_max
                                ),
                                safe_final_min=(
                                    config.scorer_formula_cf_safe_final_min
                                ),
                                progress_high_min=(
                                    config.scorer_formula_cf_progress_high_min
                                ),
                                progress_low_max=(
                                    config.scorer_formula_cf_progress_low_max
                                ),
                                yaw_weight=config.scorer_anchor_yaw_weight,
                                seed=config.scorer_anchor_seed,
                            ),
                        )
                    )
                else:
                    local_count = (
                        config.scorer_anchor_num_local_hard
                        if sampling == "hard_balanced"
                        else 0
                    )
                    balanced_count = config.scorer_anchor_num_balanced
                    if sampling == "balanced":
                        balanced_count += config.scorer_anchor_num_local_hard
                    gt_local_count = (
                        config.scorer_anchor_num_gt_local_hard
                        if sampling == "hard_balanced"
                        else 0
                    )
                    self._anchor_sampler = HardAnchorSampler(
                        self.anchors,
                        HardAnchorSamplerConfig(
                            num_local_hard=local_count,
                            num_gt_local_hard=gt_local_count,
                            num_balanced=balanced_count,
                            nearest_per_proposal=(
                                config.scorer_anchor_nearest_per_proposal
                            ),
                            nearest_per_gt=config.scorer_anchor_nearest_per_gt,
                            min_score_gap=config.scorer_anchor_min_score_gap,
                            yaw_weight=config.scorer_anchor_yaw_weight,
                            seed=config.scorer_anchor_seed,
                        ),
                    )
                exp_root = Path(os.getenv("NAVSIM_EXP_ROOT"))
                self._anchor_metric_cache_root = (
                    exp_root / "Drive-JEPA-cache" / "train_metric_cache"
                )
                self._anchor_score_root = (
                    Path(config.scorer_anchor_cache_root)
                    if config.scorer_anchor_cache_root
                    else exp_root
                    / "Drive-JEPA-cache"
                    / "anchors_scores_v1_full_fp16"
                )

        if not self._cache_data and (
            scorer_only_training or joint_continuation_training
        ):
            if not self._checkpoint_path:
                raise ValueError(
                    "checkpoint-based training requires a non-empty "
                    "checkpoint_path"
                )
            self.initialize()

    def name(self) -> str:
        """Inherited, see superclass."""
        return 'drive_jepa_perception_based_agent' 

    def set_training_epoch(self, epoch: int) -> None:
        """Set the local trainer epoch used by optional loss schedules."""
        self._training_epoch = max(int(epoch), 0)

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"]
            mapped_state_dict = {k.replace("agent._pad_model", "_pad_model"): v for k, v in state_dict.items()}
            hist_key = "_pad_model.hist_encoding.weight"
            if hist_key in mapped_state_dict:
                source_hist = mapped_state_dict[hist_key]
                target_hist = self.state_dict()[hist_key]
                if source_hist.shape != target_hist.shape:
                    if (
                        source_hist.ndim != 2
                        or target_hist.ndim != 2
                        or source_hist.shape[0] != target_hist.shape[0]
                        or source_hist.shape[1] < 7
                        or target_hist.shape[1] < 7
                    ):
                        raise RuntimeError(
                            "cannot adapt checkpoint ego-status projection "
                            f"from {tuple(source_hist.shape)} to "
                            f"{tuple(target_hist.shape)}"
                        )
                    # Preserve the seven kinematic columns and all command
                    # columns shared by NAVSIM (4) and Bench2Drive (6).  New
                    # B2D lane-change command columns retain their initialized
                    # weights and are learned during fine-tuning.
                    adapted_hist = target_hist.clone()
                    adapted_hist[:, :7] = source_hist[:, :7]
                    shared_commands = min(
                        source_hist.shape[1] - 7,
                        target_hist.shape[1] - 7,
                    )
                    adapted_hist[:, 7 : 7 + shared_commands] = source_hist[
                        :, 7 : 7 + shared_commands
                    ]
                    mapped_state_dict[hist_key] = adapted_hist
                    print(
                        "Adapted checkpoint ego-status projection: "
                        f"{tuple(source_hist.shape)} -> "
                        f"{tuple(target_hist.shape)}"
                    )
            incompatible = self.load_state_dict(mapped_state_dict, strict=False)
            allowed_unexpected = [
                k for k in incompatible.unexpected_keys
                if (
                    k.startswith("_pad_model._frozen_pretrain_vit.")
                    or k.startswith("_pad_model._ema_target_backbone.")
                )
            ]
            allowed_missing = [
                k for k in incompatible.missing_keys
                if (
                    k.startswith("_pad_model._frozen_pretrain_vit.")
                    or k.startswith("_pad_model._ema_target_backbone.")
                )
            ]
            unexpected = [
                k for k in incompatible.unexpected_keys
                if k not in allowed_unexpected
            ]
            missing = [
                k for k in incompatible.missing_keys
                if k not in allowed_missing
            ]
            if missing or unexpected:
                raise RuntimeError(
                    "Error(s) in loading state_dict for DriveJEPAAgent:\n"
                    f"\tMissing key(s): {missing}\n"
                    f"\tUnexpected key(s): {unexpected}"
                )
            if allowed_unexpected:
                print(
                    "Ignoring future target encoder checkpoint keys during eval: "
                    f"{len(allowed_unexpected)} unexpected keys"
                )
            if allowed_missing:
                print(
                    "Ignoring missing future target encoder checkpoint keys during eval: "
                    f"{len(allowed_missing)} keys"
                )

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[2, 3],
            cam_l0=[3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[3],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[3],
            lidar_pc=[],
        )
    
    def get_target_builders(self):
        return [DriveJEPATargetBuilder(config=self._config)]

    def get_feature_builders(self):
        return [DriveJEPAFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._cache_data:
            raise RuntimeError("DriveJEPAAgent was initialized with cache_data=True and cannot run forward().")
        return self._pad_model(features)

    def compute_score(self, targets, proposals, test=True):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        target_trajectory = targets["trajectory"]
        proposals=proposals.detach()

        if self.b2d:
            data_points = self.b2d_before_score(
                self.b2d_map_infos,
                proposals,
                targets,
                metric_cache_paths,
                self._config,
            )
        else:
            data_points = [
                {
                    "token": metric_cache_paths[token],
                    "poses": poses,
                    "test": test
                }
                for token, poses in zip(
                    targets["token"], proposals.float().cpu().numpy()
                )
            ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)
        scores_index = [res[-1] for res in all_res]

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)
            target_raw_progress = torch.FloatTensor(
                np.stack([res[4] for res in all_res])
            ).to(proposals.device)
            target_pdm_progress = torch.FloatTensor(
                np.asarray([res[5] for res in all_res])
            ).to(proposals.device)

            return (
                final_scores,
                best_scores,
                target_scores,
                key_agent_corners,
                key_agent_labels,
                all_ego_areas,
                scores_index,
                target_raw_progress,
                target_pdm_progress,
            )

    def score_loss(self, pred_logit, pred_logit2,agents_state, pred_area_logits, target_scores, gt_states, gt_valid,
                   gt_ego_areas):

        if agents_state is not None:
            pred_states = agents_state[..., :-1].reshape(gt_states.shape)
            pred_logits = agents_state[..., -1:].reshape(gt_valid.shape)

            pred_l1_loss = F.l1_loss(pred_states, gt_states, reduction="none")[gt_valid]

            if len(pred_l1_loss):
                pred_l1_loss = pred_l1_loss.mean()
            else:
                pred_l1_loss = pred_states.mean() * 0

            pred_ce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_valid.to(torch.float32), reduction="mean")

        else:
            pred_ce_loss = 0
            pred_l1_loss = 0

        if pred_area_logits is not None:
            pred_area_logits = pred_area_logits.reshape(gt_ego_areas.shape)

            pred_area_loss = F.binary_cross_entropy_with_logits(pred_area_logits, gt_ego_areas.to(torch.float32),
                                                              reduction="mean")
        else:
            pred_area_loss = 0

        if pred_logit.shape[-1] == 1:
            # EMA-JQTF Direct-Final: there are no unused metric heads.
            sub_score_loss = pred_logit.sum() * 0.0
        elif (
            getattr(self._config, "future_gt_trajectory_conditioned", False)
            and pred_logit.shape[-1] == 6
        ):
            # SC-3: directly supervise the five PDM metric heads here; the
            # final head is supervised exactly once by final_score_loss below.
            sub_score_loss = self.bce_logit_loss(
                pred_logit[..., :-1], target_scores[..., -6:-1]
            )
        else:
            sub_score_loss = self.bce_logit_loss(
                pred_logit, target_scores[..., -pred_logit.shape[-1]:]
            )

        final_score_loss = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])  # .mean()

        if pred_logit2 is not None:
            sub_score_loss2 = self.bce_logit_loss(pred_logit2, target_scores)  # .mean()[..., -6:-1][..., -6:-1]

            final_score_loss2 = self.bce_logit_loss(pred_logit2[..., -1], target_scores[..., -1])  # .mean()

            sub_score_loss=(sub_score_loss+sub_score_loss2)/2

            final_score_loss=(final_score_loss+final_score_loss2)/2

        return sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss

    def formula_progress_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target_scores: torch.Tensor,
        target_raw_progress: torch.Tensor,
        target_pdm_progress: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Formula loss for four metric logits and two progress heads."""
        candidate_output = pred["pred_logit"]
        pdm_progress_norm = pred["pred_pdm_progress_norm"]
        if candidate_output.shape[-1] != 5:
            raise ValueError(
                "formula_progress candidate output must be [B,P,5], got "
                f"{tuple(candidate_output.shape)}"
            )
        if pdm_progress_norm is None:
            raise ValueError(
                "formula_progress requires pred_pdm_progress_norm"
            )
        if target_raw_progress.shape != candidate_output.shape[:-1]:
            raise ValueError(
                "raw-progress target must align with candidates, got "
                f"{tuple(target_raw_progress.shape)} vs "
                f"{tuple(candidate_output.shape[:-1])}"
            )
        if tuple(pdm_progress_norm.shape) != (
            candidate_output.shape[0],
            1,
        ):
            raise ValueError(
                "PDM progress prediction must be [B,1], got "
                f"{tuple(pdm_progress_norm.shape)}"
            )

        # target_scores order is [NOC,DAC,EP,TTC,comfort,final].
        metric_target = target_scores[..., [0, 1, 3, 4]].float()
        metric_logits = candidate_output[..., :4].float()
        metric_loss_type = getattr(
            self._config,
            "scorer_formula_metric_loss_type",
            "smooth_l1",
        )
        noc_failure_weight = float(
            getattr(self._config, "scorer_formula_noc_failure_weight", 1.0)
        )
        noc_partial_weight = float(
            getattr(self._config, "scorer_formula_noc_partial_weight", 1.0)
        )
        ttc_failure_weight = float(
            getattr(self._config, "scorer_formula_ttc_failure_weight", 1.0)
        )
        if metric_loss_type == "smooth_l1":
            if (
                noc_failure_weight != 1.0
                or noc_partial_weight != 1.0
                or ttc_failure_weight != 1.0
            ):
                raise ValueError(
                    "Safety failure weighting is supported only with Formula "
                    "metric loss type 'bce'"
                )
            metric_per_head = F.smooth_l1_loss(
                torch.sigmoid(metric_logits),
                metric_target,
                reduction="none",
            ).mean(dim=(0, 1))
        elif metric_loss_type == "bce":
            metric_per_head = formula_metric_bce_per_head(
                metric_logits,
                metric_target,
                noc_failure_weight=noc_failure_weight,
                noc_partial_weight=noc_partial_weight,
                ttc_failure_weight=ttc_failure_weight,
            )
        elif metric_loss_type == "focal":
            metric_per_head = formula_metric_focal_per_head(
                metric_logits,
                metric_target,
                gamma=float(self._config.scorer_formula_focal_gamma),
                alpha=float(self._config.scorer_formula_focal_alpha),
                noc_failure_weight=noc_failure_weight,
                noc_partial_weight=noc_partial_weight,
                ttc_failure_weight=ttc_failure_weight,
            )
        else:
            raise ValueError(
                "scorer_formula_metric_loss_type must be "
                f"'smooth_l1', 'bce', or 'focal', got {metric_loss_type!r}"
            )

        progress_scale = float(
            self._config.scorer_progress_scale
        )
        raw_target_norm = (
            target_raw_progress.float().clamp_min(0.0)
            / progress_scale
        )
        pdm_target_norm = (
            target_pdm_progress.float().clamp_min(0.0)
            / progress_scale
        ).unsqueeze(-1)
        raw_progress_per_candidate = F.smooth_l1_loss(
            candidate_output[..., 4].float(),
            raw_target_norm,
            reduction="none",
        )
        progress_topk = min(
            max(int(getattr(self._config, "scorer_progress_topk", 8)), 1),
            target_scores.shape[1],
        )
        progress_topk_weight = float(
            getattr(self._config, "scorer_progress_topk_weight", 1.0)
        )
        if progress_topk_weight <= 0.0:
            raise ValueError("scorer_progress_topk_weight must be positive")
        raw_progress_weights = torch.ones_like(raw_progress_per_candidate)
        if progress_topk_weight != 1.0:
            progress_topk_index = torch.topk(
                target_scores[..., 2].float(),
                k=progress_topk,
                dim=1,
                largest=True,
                sorted=False,
            ).indices
            raw_progress_weights.scatter_(
                1, progress_topk_index, progress_topk_weight
            )
        raw_progress_loss = (
            raw_progress_per_candidate * raw_progress_weights
        ).sum() / raw_progress_weights.sum().clamp_min(1.0)
        # This is deliberately evaluated once per scene.  The target is not
        # broadcast across proposals, so its effective weight cannot grow
        # with candidate or counterfactual count.
        pdm_progress_loss = F.smooth_l1_loss(
            pdm_progress_norm.float(),
            pdm_target_norm,
        )
        raw_progress_loss_weight = float(
            getattr(
                self._config,
                "scorer_formula_raw_progress_loss_weight",
                1.0,
            )
        )
        pdm_progress_loss_weight = float(
            getattr(
                self._config,
                "scorer_formula_pdm_progress_loss_weight",
                1.0,
            )
        )
        if raw_progress_loss_weight < 0.0 or pdm_progress_loss_weight < 0.0:
            raise ValueError("Formula Progress loss weights must be non-negative")
        # Keep the legacy /6 normalization fixed. Increasing Progress weight
        # adds Progress gradient without silently reducing the four safety-head
        # gradient coefficients.
        scorer_loss = (
            metric_per_head.sum()
            + raw_progress_loss_weight * raw_progress_loss
            + pdm_progress_loss_weight * pdm_progress_loss
        ) / 6.0

        pred_raw_meter = (
            candidate_output[..., 4].float() * progress_scale
        )
        pred_pdm_meter = (
            pdm_progress_norm.float().squeeze(-1)
            * progress_scale
        )
        return {
            "scorer_loss": scorer_loss,
            "metric_loss": metric_per_head.mean(),
            "noc_loss": metric_per_head[0],
            "dac_loss": metric_per_head[1],
            "ttc_loss": metric_per_head[2],
            "comfort_loss": metric_per_head[3],
            "raw_progress_loss": raw_progress_loss,
            "pdm_progress_loss": pdm_progress_loss,
            "raw_progress_mae_m": F.l1_loss(
                pred_raw_meter, target_raw_progress.float()
            ),
            "pdm_progress_mae_m": F.l1_loss(
                pred_pdm_meter, target_pdm_progress.float()
            ),
            "ep_mae": F.l1_loss(
                pred["pred_ep"].float(),
                target_scores[..., 2].float(),
            ),
            "formula_mae": F.l1_loss(
                pred["pdm_score"].float(),
                target_scores[..., -1].float(),
            ),
        }

    def diversity_loss(self, proposals):
        dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)

        dist = dist + (dist == 0)

        #dist[dist==0]=10000

        inter_loss = -dist.amin(1).amin(1).mean()

        return inter_loss
    
    def trajectory_loss_anchors(self, proposal_list, target_trajectory, config, scores_index): 
        trajectory_loss = 0

        min_loss_list = []
        inter_loss_list = []
        for proposals_i, idx_arr in zip(proposal_list, scores_index):
            min_loss = (
                torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1)
                .mean(-1)
                .amin(1)
                .mean()
            )

            pseudo_min_loss = 0 
            # Use scores_index to build pseudo targets from anchors
            for i, idx_arr in enumerate(scores_index):
                if idx_arr.shape[0] == 0:
                    continue

                if idx_arr.shape[0] > 4:
                    sampled_idx = np.random.choice(idx_arr, size=4, replace=False)
                else:
                    sampled_idx = idx_arr

                # self.anchors: numpy; convert to torch on same device/dtype as proposals_i
                anchors_np = self.anchors[sampled_idx]
                pseudo_targets = torch.from_numpy(anchors_np).to(
                    device=proposals_i.device, dtype=proposals_i.dtype
                )

                # proposals_i: (B, P, T, D)
                # pseudo_targets: (K, T, D)
                diff = proposals_i[i, None] - pseudo_targets[:, None]  # (B, K, P, T, D)
                pseudo_min_loss += (
                    torch.linalg.norm(diff, dim=-1, ord=1)   # (B, K, P, T)
                    .mean(-1)                                # (B, K, P)
                    .amin(-1)                                # (B, K)
                    .mean()                                  # scalar
                )
            
            min_loss += 0.5 * pseudo_min_loss / len(scores_index) 

            inter_loss = self.diversity_loss(proposals_i)
            trajectory_loss = config.prev_weight * trajectory_loss + min_loss + inter_loss * config.inter_weight

            min_loss_list.append(min_loss)
            inter_loss_list.append(inter_loss)

        return trajectory_loss, min_loss, inter_loss, min_loss_list, inter_loss_list

    def scorer_ranking_loss(
        self,
        pred_logit,
        target_final,
        margin,
        pred_reward=None,
    ):
        """Pairwise margin ranking on the score used for inference selection.

        ``pred_logit`` is either the six factorized legacy logits or the
        EMA-JQTF Direct-Final logit ``[B,P,1]``.
        target_final: [B, P] target PDM final score
        """
        if pred_reward is not None:
            if pred_reward.shape != target_final.shape:
                raise ValueError(
                    "pred_reward and target_final must align"
                )
        elif pred_logit.shape[-1] == 1:
            pred_reward = torch.sigmoid(pred_logit[..., 0])
        else:
            s = torch.sigmoid(pred_logit)
            noc, dac, ep, ttc, comfort = (
                s[..., 0],
                s[..., 1],
                s[..., 2],
                s[..., 3],
                s[..., 4],
            )
            pred_reward = (
                noc
                * dac
                * (5.0 * ttc + 5.0 * ep + 2.0 * comfort)
                / 12.0
            )
        diff = pred_reward[:, :, None] - pred_reward[:, None, :]                 # [B, P, P]
        tgt = torch.sign(target_final[:, :, None] - target_final[:, None, :])   # [B, P, P]
        loss = torch.relu(margin - tgt * diff)
        mask = tgt != 0
        denom = mask.sum().clamp(min=1)
        return (loss * mask).sum() / denom

    def scorer_progress_ranking_loss(
        self,
        pred_progress: torch.Tensor,
        target_progress: torch.Tensor,
    ):
        """Rank target EP scores, emphasizing pairs led by a true top-k item.

        Safety is intentionally not used as a filter: safety heads supervise
        every proposal, while this loss learns the independent Progress
        factor later consumed by the multiplicative Formula score.
        """
        output = progress_ranking_loss(
            pred_progress,
            target_progress,
            topk=int(getattr(self._config, "scorer_progress_topk", 8)),
            topk_weight=float(
                getattr(self._config, "scorer_progress_topk_weight", 1.0)
            ),
            margin_cap=float(
                getattr(
                    self._config,
                    "scorer_progress_ranking_margin_cap",
                    0.05,
                )
            ),
            tie_epsilon=float(
                getattr(
                    self._config,
                    "scorer_progress_ranking_tie_epsilon",
                    1e-4,
                )
            ),
        )
        return output.loss, output.pair_count, output.accuracy

    def scorer_asymmetric_safety_loss(
        self,
        pred_logit: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> AsymmetricSafetyLossOutput:
        """Directional final-score loss shared by generated and anchor paths."""

        return asymmetric_safety_loss(
            pred_logit[..., -1],
            target_scores,
            false_safe_weight=self._config.scorer_false_safe_weight,
            false_unsafe_weight=self._config.scorer_false_unsafe_weight,
            error_margin=self._config.scorer_asymmetric_safety_margin,
            safety_metric_threshold=self._config.scorer_safety_metric_threshold,
            safe_final_threshold=self._config.scorer_safe_final_threshold,
        )

    def _anchor_scene_seed(self, token: Any) -> int:
        digest = hashlib.sha256(str(token).encode("utf-8")).digest()
        return self._config.scorer_anchor_seed ^ int.from_bytes(digest[:4], "little")

    def formula_counterfactual_auxiliary_loss(
        self,
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Point-supervise Formula heads on a fixed 128-anchor curriculum.

        Anchor caches contain [NOC,DAC,EP,TTC,Comfort,final], but do not
        contain raw-progress meters. Consequently the external path directly
        supervises the four metric logits and the derived EP only on truly
        safe anchors. The scene-level PDM-progress target remains exclusively
        supervised by the native generated-candidate Formula loss.
        """
        if not hasattr(self, "_formula_counterfactual_sampler"):
            raise RuntimeError(
                "Formula counterfactual sampler was not initialized"
            )
        required = (
            "proposal_pose_features",
            "projected_map",
            "future_context_latent",
            "pred_pdm_progress_norm",
        )
        missing = [name for name in required if name not in pred]
        if missing:
            raise RuntimeError(
                f"Formula counterfactual path missing outputs: {missing}"
            )

        proposals_np = pred["proposals"].detach().float().cpu().numpy()
        sampled_trajectories = []
        sampled_targets = []
        nearest_proposals = []
        sampled_distances = []
        bucket_counts = {
            "ttc_only": [],
            "ttc_noc_joint": [],
            "safe_rescue": [],
            "safe_progress": [],
            "balanced": [],
        }
        metric_cache_paths = (
            self.train_metric_cache_paths
            if self.training
            else self.test_metric_cache_paths
        )
        for batch_index, token in enumerate(targets["token"]):
            metric_path = Path(metric_cache_paths[token])
            score_path = anchor_score_cache_path(
                metric_path,
                self._anchor_metric_cache_root,
                self._anchor_score_root,
            )
            anchor_scores = load_anchor_scores(score_path)
            sample = self._formula_counterfactual_sampler.sample(
                proposals_np[batch_index],
                anchor_scores,
                seed=self._anchor_scene_seed(token),
            )
            sampled_trajectories.append(self.anchors[sample.indices])
            sampled_targets.append(
                np.asarray(anchor_scores[sample.indices], dtype=np.float32)
            )
            nearest_proposals.append(sample.nearest_proposal_indices)
            sampled_distances.append(sample.distances)
            bucket_counts["ttc_only"].append(
                int(np.sum(sample.buckets == "ttc_only"))
            )
            bucket_counts["ttc_noc_joint"].append(
                int(np.sum(sample.buckets == "ttc_noc_joint"))
            )
            bucket_counts["safe_rescue"].append(
                int(np.sum(sample.buckets == "safe_rescue"))
            )
            bucket_counts["safe_progress"].append(
                int(
                    np.sum(
                        np.char.startswith(
                            sample.buckets.astype(str), "safe_progress_"
                        )
                    )
                )
            )
            bucket_counts["balanced"].append(
                int(
                    np.sum(
                        np.char.startswith(
                            sample.buckets.astype(str), "balanced_"
                        )
                    )
                )
            )

        device = pred["proposal_pose_features"].device
        dtype = pred["proposal_pose_features"].dtype
        anchors = torch.as_tensor(
            np.stack(sampled_trajectories), device=device, dtype=dtype
        )
        anchor_targets = torch.as_tensor(
            np.stack(sampled_targets), device=device, dtype=torch.float32
        )
        nearest_indices = torch.as_tensor(
            np.stack(nearest_proposals), device=device, dtype=torch.long
        )
        pose_features = pred["proposal_pose_features"]
        gather_indices = nearest_indices[..., None, None].expand(
            -1, -1, pose_features.shape[2], pose_features.shape[3]
        )
        nearest_features = torch.gather(
            pose_features, dim=1, index=gather_indices
        ).detach()
        anchor_output = self._pad_model.score_external_trajectories(
            anchors,
            nearest_features,
            pred["projected_map"],
            pred["future_context_latent"],
        )
        if anchor_output.shape[-1] != 5:
            raise RuntimeError(
                "Formula counterfactual scorer must output [B,A,5], got "
                f"{tuple(anchor_output.shape)}"
            )

        metric_target = anchor_targets[..., [0, 1, 3, 4]]
        metric_loss_type = getattr(
            self._config, "scorer_formula_metric_loss_type", "smooth_l1"
        )
        if metric_loss_type == "focal":
            metric_per_head = formula_metric_focal_per_head(
                anchor_output[..., :4].float(),
                metric_target,
                gamma=float(self._config.scorer_formula_focal_gamma),
                alpha=float(self._config.scorer_formula_focal_alpha),
                noc_failure_weight=float(
                    getattr(
                        self._config,
                        "scorer_formula_noc_failure_weight",
                        1.0,
                    )
                ),
                noc_partial_weight=float(
                    getattr(
                        self._config,
                        "scorer_formula_noc_partial_weight",
                        1.0,
                    )
                ),
                ttc_failure_weight=float(
                    getattr(
                        self._config,
                        "scorer_formula_ttc_failure_weight",
                        1.0,
                    )
                ),
            )
        else:
            metric_per_head = formula_metric_bce_per_head(
                anchor_output[..., :4].float(),
                metric_target,
                noc_failure_weight=float(
                    getattr(
                        self._config,
                        "scorer_formula_noc_failure_weight",
                        1.0,
                    )
                ),
                noc_partial_weight=float(
                    getattr(
                        self._config,
                        "scorer_formula_noc_partial_weight",
                        1.0,
                    )
                ),
                ttc_failure_weight=float(
                    getattr(
                        self._config,
                        "scorer_formula_ttc_failure_weight",
                        1.0,
                    )
                ),
            )

        metric_scores = torch.sigmoid(anchor_output[..., :4].float())
        formula = formula_progress_selection(
            metric_scores,
            anchor_output[..., 4].float(),
            pred["pred_pdm_progress_norm"].float().detach(),
            progress_scale=float(self._config.scorer_progress_scale),
            mode="current",
            safety_topk=int(self._config.scorer_formula_safety_topk),
            progress_mode=self._config.scorer_formula_progress_mode,
            progress_distance_threshold=float(
                self._config.scorer_progress_distance_threshold
            ),
        )
        safe_threshold = float(
            self._config.scorer_formula_cf_safe_metric_min
        )
        safe_mask = (
            (anchor_targets[..., 0] >= safe_threshold)
            & (anchor_targets[..., 1] >= safe_threshold)
            & (anchor_targets[..., 3] >= safe_threshold)
        )
        if bool(safe_mask.any()):
            ep_loss = F.smooth_l1_loss(
                formula.ego_progress[safe_mask],
                anchor_targets[..., 2][safe_mask],
            )
        else:
            ep_loss = anchor_output.sum() * 0.0
        total = (metric_per_head.sum() + ep_loss) / 5.0

        pred_safe = (
            (metric_scores[..., 0] >= 0.5)
            & (metric_scores[..., 1] >= 0.5)
            & (metric_scores[..., 2] >= 0.5)
        )
        true_unsafe = ~safe_mask
        false_safe_rate = (
            (pred_safe & true_unsafe).float().sum()
            / true_unsafe.float().sum().clamp_min(1.0)
        )
        distances = torch.as_tensor(
            np.stack(sampled_distances),
            device=device,
            dtype=torch.float32,
        )
        result = {
            "loss": total,
            "noc_loss": metric_per_head[0],
            "dac_loss": metric_per_head[1],
            "ttc_loss": metric_per_head[2],
            "comfort_loss": metric_per_head[3],
            "ep_loss": ep_loss,
            "formula_mae": F.l1_loss(
                formula.formula_score, anchor_targets[..., 5]
            ),
            "false_safe_rate": false_safe_rate,
            "distance": distances.mean(),
        }
        for name, counts in bucket_counts.items():
            result[f"{name}_count"] = torch.as_tensor(
                counts, device=device, dtype=torch.float32
            ).mean()
        return result

    def anchor_auxiliary_loss(
        self,
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
        target_scores: torch.Tensor,
        gt_pdms: Optional[torch.Tensor] = None,
    ):
        """Sample cached anchors and compute training-only pointwise scorer BCE."""
        required = (
            "proposal_pose_features",
            "projected_map",
            "future_context_latent",
        )
        missing = [name for name in required if name not in pred]
        if missing:
            raise RuntimeError(
                f"anchor auxiliary path missing model outputs: {missing}"
            )

        proposals_np = pred["proposals"].detach().float().cpu().numpy()
        generated_pdms_np = target_scores[..., -1].detach().float().cpu().numpy()
        gt_trajectories_np = targets["trajectory"].detach().float().cpu().numpy()
        gt_pdms_np = (
            None if gt_pdms is None else gt_pdms.detach().float().cpu().numpy()
        )
        if self._anchor_sampler.config.num_gt_local_hard > 0 and gt_pdms_np is None:
            raise RuntimeError("E3 anchor sampling requires online GT PDMS")
        sampled_trajectories = []
        sampled_targets = []
        nearest_proposals = []
        sampled_distances = []
        sampled_gaps = []
        sampled_hard_masks = []
        sampled_gt_hard_masks = []
        metric_cache_paths = (
            self.train_metric_cache_paths if self.training else self.test_metric_cache_paths
        )
        for batch_index, token in enumerate(targets["token"]):
            metric_path = Path(metric_cache_paths[token])
            score_path = anchor_score_cache_path(
                metric_path,
                self._anchor_metric_cache_root,
                self._anchor_score_root,
            )
            anchor_scores = load_anchor_scores(score_path)
            sample = self._anchor_sampler.sample(
                proposals_np[batch_index],
                generated_pdms_np[batch_index],
                anchor_scores,
                seed=self._anchor_scene_seed(token),
                gt_trajectory=gt_trajectories_np[batch_index],
                gt_pdms=(
                    None if gt_pdms_np is None else gt_pdms_np[batch_index]
                ),
            )
            sampled_trajectories.append(self.anchors[sample.indices])
            sampled_targets.append(
                np.asarray(anchor_scores[sample.indices], dtype=np.float32)
            )
            nearest_proposals.append(sample.nearest_proposal_indices)
            sampled_distances.append(sample.distances)
            sampled_gaps.append(sample.score_gaps)
            sampled_hard_masks.append(sample.is_local_hard)
            sampled_gt_hard_masks.append(sample.is_gt_local_hard)

        device = pred["proposal_pose_features"].device
        dtype = pred["proposal_pose_features"].dtype
        anchors = torch.as_tensor(
            np.stack(sampled_trajectories), device=device, dtype=dtype
        )
        anchor_targets = torch.as_tensor(
            np.stack(sampled_targets), device=device, dtype=torch.float32
        )
        nearest_indices = torch.as_tensor(
            np.stack(nearest_proposals), device=device, dtype=torch.long
        )
        pose_features = pred["proposal_pose_features"]
        gather_indices = nearest_indices[..., None, None].expand(
            -1, -1, pose_features.shape[2], pose_features.shape[3]
        )
        nearest_features = torch.gather(
            pose_features, dim=1, index=gather_indices
        ).detach()
        anchor_logits = self._pad_model.score_external_trajectories(
            anchors,
            nearest_features,
            pred["projected_map"],
            pred["future_context_latent"],
        )
        if anchor_logits.shape[-1] == 1:
            anchor_subscore_loss = anchor_logits.sum() * 0.0
            anchor_final_loss = F.binary_cross_entropy_with_logits(
                anchor_logits[..., 0].float(), anchor_targets[..., -1]
            )
        else:
            anchor_subscore_loss = F.binary_cross_entropy_with_logits(
                anchor_logits[..., :5].float(), anchor_targets[..., :5]
            )
            anchor_final_loss = F.binary_cross_entropy_with_logits(
                anchor_logits[..., 5].float(), anchor_targets[..., 5]
            )
        anchor_asymmetric_safety = self.scorer_asymmetric_safety_loss(
            anchor_logits, anchor_targets
        )

        distances = torch.as_tensor(
            np.stack(sampled_distances), device=device, dtype=torch.float32
        )
        score_gaps = torch.as_tensor(
            np.stack(sampled_gaps), device=device, dtype=torch.float32
        )
        hard_mask = torch.as_tensor(
            np.stack(sampled_hard_masks), device=device, dtype=torch.bool
        )
        gt_hard_mask = torch.as_tensor(
            np.stack(sampled_gt_hard_masks), device=device, dtype=torch.bool
        )
        mean_gt_hard_count = gt_hard_mask.sum(dim=1).float().mean()
        if bool(hard_mask.any()):
            mean_distance = distances[hard_mask].mean()
            mean_gap = score_gaps[hard_mask].mean()
        else:
            mean_distance = distances.mean()
            mean_gap = score_gaps.mean()
        return (
            anchor_subscore_loss,
            anchor_final_loss,
            anchor_asymmetric_safety,
            mean_distance,
            mean_gap,
            mean_gt_hard_count,
        )

    def pad_loss(self, targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config,
                 features: Dict[str, torch.Tensor] = None):

        proposals = pred["proposals"]
        proposal_list = pred["proposal_list"]
        target_trajectory = targets["trajectory"]

        use_gt_anchor_sampling = (
            getattr(config, "scorer_anchor_auxiliary", False)
            and getattr(config, "scorer_anchor_num_gt_local_hard", 0) > 0
        )
        score_trajectories = (
            torch.cat([proposals, target_trajectory[:, None]], dim=1)
            if use_gt_anchor_sampling
            else proposals
        )
        (
            final_scores,
            best_scores,
            target_scores,
            gt_states,
            gt_valid,
            gt_ego_areas,
            scores_index,
            target_raw_progress,
            target_pdm_progress,
        ) = self.compute_score(
            targets, score_trajectories, test=False
        )
        anchor_gt_pdms = None
        if use_gt_anchor_sampling:
            proposal_count = proposals.shape[1]
            anchor_gt_pdms = target_scores[:, proposal_count, -1]
            final_scores = final_scores[:, :proposal_count]
            best_scores = torch.amax(final_scores, dim=-1)
            target_scores = target_scores[:, :proposal_count]
            gt_states = gt_states[:, :proposal_count]
            gt_valid = gt_valid[:, :proposal_count]
            gt_ego_areas = gt_ego_areas[:, :proposal_count]
            target_raw_progress = target_raw_progress[
                :, :proposal_count
            ]

        trajectory_loss, min_loss, inter_loss, min_loss_list, inter_loss_list = self.trajectory_loss_anchors(proposal_list, target_trajectory, config, scores_index)

        min_loss0 = min_loss_list[0]
        inter_loss0 = inter_loss_list[0]
        # min_loss1 = min_loss_list[1]
        # inter_loss1 = inter_loss_list[1]

        is_formula_progress = (
            getattr(config, "future_integration", None) == "ema_jqtf"
            and getattr(
                config, "ema_jqtf_scorer_head_mode", None
            )
            == "formula_progress"
        )
        formula_zero = pred["pred_logit"].sum() * 0.0
        formula_losses = {
            "metric_loss": formula_zero,
            "noc_loss": formula_zero,
            "dac_loss": formula_zero,
            "ttc_loss": formula_zero,
            "comfort_loss": formula_zero,
            "raw_progress_loss": formula_zero,
            "pdm_progress_loss": formula_zero,
            "raw_progress_mae_m": formula_zero.detach(),
            "pdm_progress_mae_m": formula_zero.detach(),
            "ep_mae": formula_zero.detach(),
            "formula_mae": formula_zero.detach(),
        }
        if is_formula_progress:
            formula_losses = self.formula_progress_loss(
                pred,
                target_scores,
                target_raw_progress,
                target_pdm_progress,
            )
            sub_score_loss = formula_losses["scorer_loss"]
            final_score_loss = formula_zero
            pred_ce_loss = formula_zero
            pred_l1_loss = formula_zero
            pred_area_loss = formula_zero
        elif "pred_logit" in pred.keys():
            sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
                pred["pred_logit"],pred["pred_logit2"],
                pred["pred_agents_states"], pred["pred_area_logit"]
                , target_scores, gt_states, gt_valid, gt_ego_areas)
        else:
            sub_score_loss = final_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0

        safety_logits = pred["pred_logit"]
        if is_formula_progress:
            safety_logits = torch.logit(
                pred["pdm_score"].clamp(1e-6, 1.0 - 1e-6)
            ).unsqueeze(-1)
        generated_asymmetric_safety = self.scorer_asymmetric_safety_loss(
            safety_logits, target_scores
        )
        anchor_zero = pred["pred_logit"].sum() * 0.0
        anchor_subscore_loss = anchor_zero
        anchor_final_loss = anchor_zero
        anchor_asymmetric_safety_loss = anchor_zero
        anchor_false_safe_loss = anchor_zero
        anchor_false_unsafe_loss = anchor_zero
        anchor_false_safe_rate = anchor_zero.detach()
        anchor_false_unsafe_rate = anchor_zero.detach()
        anchor_unsafe_fraction = anchor_zero.detach()
        anchor_safe_high_fraction = anchor_zero.detach()
        anchor_distance = anchor_zero.detach()
        anchor_score_gap = anchor_zero.detach()
        anchor_gt_local_hard_count = anchor_zero.detach()
        formula_cf = {
            "noc_loss": anchor_zero,
            "dac_loss": anchor_zero,
            "ttc_loss": anchor_zero,
            "comfort_loss": anchor_zero,
            "ep_loss": anchor_zero,
            "formula_mae": anchor_zero.detach(),
            "false_safe_rate": anchor_zero.detach(),
            "ttc_only_count": anchor_zero.detach(),
            "ttc_noc_joint_count": anchor_zero.detach(),
            "safe_rescue_count": anchor_zero.detach(),
            "safe_progress_count": anchor_zero.detach(),
            "balanced_count": anchor_zero.detach(),
        }
        if getattr(config, "scorer_anchor_auxiliary", False):
            if is_formula_progress and getattr(
                config, "scorer_formula_counterfactual", False
            ):
                formula_cf = self.formula_counterfactual_auxiliary_loss(
                    targets, pred
                )
                anchor_subscore_loss = formula_cf["loss"]
                anchor_distance = formula_cf["distance"]
                anchor_false_safe_rate = formula_cf["false_safe_rate"]
            else:
                (
                    anchor_subscore_loss,
                    anchor_final_loss,
                    anchor_asymmetric_safety,
                    anchor_distance,
                    anchor_score_gap,
                    anchor_gt_local_hard_count,
                ) = self.anchor_auxiliary_loss(
                    targets, pred, target_scores, gt_pdms=anchor_gt_pdms
                )
                anchor_asymmetric_safety_loss = anchor_asymmetric_safety.total
                anchor_false_safe_loss = anchor_asymmetric_safety.false_safe
                anchor_false_unsafe_loss = anchor_asymmetric_safety.false_unsafe
                anchor_false_safe_rate = anchor_asymmetric_safety.false_safe_rate
                anchor_false_unsafe_rate = anchor_asymmetric_safety.false_unsafe_rate
                anchor_unsafe_fraction = anchor_asymmetric_safety.unsafe_fraction
                anchor_safe_high_fraction = (
                    anchor_asymmetric_safety.safe_high_fraction
                )

        if pred["agent_states"] is not None:
            agent_class_loss, agent_box_loss = _agent_loss(targets, pred, config)
        else:
            agent_class_loss = 0
            agent_box_loss = 0

        if pred["bev_semantic_map"] is not None:
            bev_semantic_loss = F.cross_entropy(pred["bev_semantic_map"], targets["bev_semantic_map"].long())
        else:
            bev_semantic_loss = 0

        scheduled_scorer_weight = scheduled_loss_weight(
            getattr(config, "scorer_loss_weight_schedule", ()),
            self._training_epoch,
            config.sub_score_weight,
        )
        scheduled_future_weight = scheduled_loss_weight(
            getattr(config, "future_prediction_weight_schedule", ()),
            self._training_epoch,
            config.future_prediction_weight,
        )
        effective_progress_ranking_weight = bundled_loss_weight(
            getattr(config, "scorer_progress_ranking_weight", 0.0),
            scheduled_scorer_weight,
            getattr(config, "scorer_progress_ranking_in_bundle", False),
        )
        effective_ranking_weight = bundled_loss_weight(
            getattr(config, "scorer_ranking_weight", 0.0),
            scheduled_scorer_weight,
            getattr(config, "scorer_ranking_in_bundle", False),
        )
        effective_safe_progress_tie_ranking_weight = bundled_loss_weight(
            getattr(config, "scorer_safe_progress_tie_ranking_weight", 0.0),
            scheduled_scorer_weight,
            getattr(
                config,
                "scorer_safe_progress_tie_ranking_in_bundle",
                False,
            ),
        )

        # === Future latent prediction loss ===
        future_loss = 0
        has_future_prediction = (
            "predicted_future_latents" in pred
            or (
                getattr(config, "future_gt_trajectory_conditioned", False)
                and "future_context_latent" in pred
            )
        )
        if (config.predict_future
                and not getattr(config, "scorer_only_training", False)
                and scheduled_future_weight > 0.0
                and has_future_prediction
                and "future_camera_features" in targets):
            train_offsets = tuple(int(x) for x in config.future_frame_offsets)
            future_cams = select_future_camera_features_for_training(targets, train_offsets)
            is_ema_jqtf = config.future_integration == "ema_jqtf"
            matched_index = None
            matched_distance = None
            if is_ema_jqtf:
                (
                    predicted_list,
                    matched_index,
                    matched_distance,
                ) = self._pad_model.predict_matched_future_latents(
                    pred["future_context_latent"],
                    pred["candidate_future_tokens"],
                    proposals,
                    target_trajectory,
                )
            elif getattr(config, "future_gt_trajectory_conditioned", False):
                # Only the observed GT trajectory has a valid future-latent
                # target. Counterfactual proposals receive scorer losses only.
                predicted_list = self._pad_model.predict_gt_future_latents(
                    pred["future_context_latent"], target_trajectory
                )
            else:
                predicted_list = pred["predicted_future_latents"]
            num_offsets = len(predicted_list)
            for i in range(num_offsets):
                with torch.no_grad():
                    frame = self._pad_model._camera_to_float(future_cams[:, i])
                    # [B, 3, H, W], [0,1] after legacy/uint8 normalization.
                    frame = self._pad_model.transform(frame)  # ImageNet normalize
                    if config.future_target_pair_mode == "current_future":
                        if "future_target_current_frame" not in pred:
                            raise RuntimeError(
                                "current_future target requires normalized frame "
                                "t from model.forward"
                            )
                        current_frame = pred["future_target_current_frame"]
                    else:
                        current_frame = frame
                    fut_input = build_future_target_camera_pair(
                        current_frame,
                        frame,
                        config.future_target_pair_mode,
                    )
                    target_latent = self._pad_model.encode_future_target(fut_input)
                    if config.future_target_layernorm:
                        batch_size, channels, height, width = target_latent.shape
                        target_latent = F.layer_norm(
                            target_latent.permute(0, 2, 3, 1),
                            (channels,),
                        ).permute(0, 3, 1, 2)
                pred_latent = predicted_list[i]
                if config.future_target_layernorm and is_ema_jqtf:
                    channels = pred_latent.shape[1]
                    pred_latent = F.layer_norm(
                        pred_latent.permute(0, 2, 3, 1),
                        (channels,),
                    ).permute(0, 3, 1, 2)
                if config.future_prediction_type == "l1":
                    future_loss = future_loss + F.l1_loss(pred_latent, target_latent)
                elif config.future_prediction_type == "cosine":
                    pred_flat = pred_latent.flatten(2)
                    tgt_flat = target_latent.flatten(2)
                    future_loss = future_loss + (
                        1 - F.cosine_similarity(pred_flat, tgt_flat, dim=1)
                    ).mean()
                else:
                    future_loss = future_loss + F.mse_loss(pred_latent, target_latent)
            if num_offsets > 0:
                future_loss = future_loss / num_offsets
            if matched_index is not None:
                pred["future_matched_index"] = matched_index
                pred["future_matched_distance"] = matched_distance.detach()

        # === SC-2: trajectory-conditioned scorer pairwise ranking loss ===
        ranking_loss = 0
        if (getattr(config, "scorer_traj_conditioned", False)
                and getattr(config, "scorer_ranking_weight", 0) > 0
                and "pred_logit" in pred.keys()):
            ranking_loss = self.scorer_ranking_loss(
                pred["pred_logit"],
                target_scores[..., -1],
                config.scorer_ranking_margin,
                pred_reward=(
                    pred["pdm_score"]
                    if is_formula_progress
                    else None
                ),
            )

        progress_ranking_loss = formula_zero
        progress_ranking_pair_count = formula_zero.detach()
        progress_ranking_accuracy = formula_zero.detach()
        if (
            is_formula_progress
            and getattr(config, "scorer_progress_ranking_weight", 0) > 0
        ):
            (
                progress_ranking_loss,
                progress_ranking_pair_count,
                progress_ranking_accuracy,
            ) = self.scorer_progress_ranking_loss(
                pred["pred_ep"],
                target_scores[..., 2],
            )

        safe_progress_tie_ranking_loss = formula_zero
        safe_progress_tie_pair_count = formula_zero.detach()
        safe_progress_tie_ranking_accuracy = formula_zero.detach()
        if (
            is_formula_progress
            and getattr(
                config, "scorer_safe_progress_tie_ranking_weight", 0.0
            )
            > 0.0
        ):
            safe_tie_output = safe_raw_progress_tie_ranking_loss(
                pred["pred_raw_progress_norm"],
                target_raw_progress.float()
                / float(config.scorer_progress_scale),
                target_scores,
                topk=int(getattr(config, "scorer_progress_topk", 8)),
                topk_weight=float(
                    getattr(config, "scorer_progress_topk_weight", 1.0)
                ),
                safety_threshold=float(
                    getattr(
                        config,
                        "scorer_safe_progress_tie_safety_threshold",
                        0.95,
                    )
                ),
                ep_tie_epsilon=float(
                    getattr(
                        config,
                        "scorer_safe_progress_tie_ep_epsilon",
                        1e-4,
                    )
                ),
                raw_tie_epsilon=float(
                    getattr(
                        config,
                        "scorer_safe_progress_tie_raw_epsilon_m",
                        0.01,
                    )
                    / float(config.scorer_progress_scale)
                ),
                margin_cap=float(
                    getattr(
                        config,
                        "scorer_progress_ranking_margin_cap",
                        0.05,
                    )
                ),
            )
            safe_progress_tie_ranking_loss = safe_tie_output.loss
            safe_progress_tie_pair_count = safe_tie_output.pair_count
            safe_progress_tie_ranking_accuracy = safe_tie_output.accuracy

        loss = (
                config.trajectory_weight * trajectory_loss
                + scheduled_scorer_weight * sub_score_loss
                + config.final_score_weight * final_score_loss
                + config.pred_ce_weight * pred_ce_loss
                + config.pred_l1_weight * pred_l1_loss
                + config.pred_area_weight * pred_area_loss
                + config.agent_class_weight * agent_class_loss
                + config.agent_box_weight * agent_box_loss
                + config.bev_semantic_weight * bev_semantic_loss
                + scheduled_future_weight * future_loss
                + effective_ranking_weight * ranking_loss
                + effective_progress_ranking_weight * progress_ranking_loss
                + effective_safe_progress_tie_ranking_weight
                * safe_progress_tie_ranking_loss
                + getattr(config, "scorer_asymmetric_safety_weight", 0)
                * (
                    generated_asymmetric_safety.total
                    + getattr(config, "scorer_anchor_loss_weight", 0)
                    * getattr(config, "scorer_anchor_final_weight", 1.0)
                    * anchor_asymmetric_safety_loss
                )
                + getattr(config, "scorer_anchor_loss_weight", 0)
                * (
                    getattr(config, "scorer_anchor_subscore_weight", 1.0)
                    * anchor_subscore_loss
                    + getattr(config, "scorer_anchor_final_weight", 1.0)
                    * anchor_final_loss
                )

        )

        pdm_score = pred["pdm_score"].detach()
        top_proposals = torch.argmax(pdm_score, dim=1)
        score = final_scores[np.arange(len(final_scores)), top_proposals].mean()
        best_score = best_scores.mean()

        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            'sub_score_loss': sub_score_loss,
            'final_score_loss': final_score_loss,
            'generated_subscore_loss': sub_score_loss,
            'generated_final_loss': final_score_loss,
            'asymmetric_safety_loss': generated_asymmetric_safety.total,
            'false_safe_loss': generated_asymmetric_safety.false_safe,
            'false_unsafe_loss': generated_asymmetric_safety.false_unsafe,
            'false_safe_rate': generated_asymmetric_safety.false_safe_rate,
            'false_unsafe_rate': generated_asymmetric_safety.false_unsafe_rate,
            'unsafe_candidate_fraction': (
                generated_asymmetric_safety.unsafe_fraction
            ),
            'safe_high_candidate_fraction': (
                generated_asymmetric_safety.safe_high_fraction
            ),
            'anchor_subscore_loss': anchor_subscore_loss,
            'anchor_final_loss': anchor_final_loss,
            'anchor_asymmetric_safety_loss': anchor_asymmetric_safety_loss,
            'anchor_false_safe_loss': anchor_false_safe_loss,
            'anchor_false_unsafe_loss': anchor_false_unsafe_loss,
            'anchor_false_safe_rate': anchor_false_safe_rate,
            'anchor_false_unsafe_rate': anchor_false_unsafe_rate,
            'anchor_unsafe_fraction': anchor_unsafe_fraction,
            'anchor_safe_high_fraction': anchor_safe_high_fraction,
            'anchor_distance': anchor_distance,
            'anchor_score_gap': anchor_score_gap,
            'anchor_gt_local_hard_count': anchor_gt_local_hard_count,
            'formula_cf_noc_loss': formula_cf["noc_loss"],
            'formula_cf_dac_loss': formula_cf["dac_loss"],
            'formula_cf_ttc_loss': formula_cf["ttc_loss"],
            'formula_cf_comfort_loss': formula_cf["comfort_loss"],
            'formula_cf_ep_loss': formula_cf["ep_loss"],
            'formula_cf_formula_mae': formula_cf["formula_mae"],
            'formula_cf_false_safe_rate': formula_cf[
                "false_safe_rate"
            ],
            'formula_cf_ttc_only_count': formula_cf["ttc_only_count"],
            'formula_cf_ttc_noc_joint_count': formula_cf[
                "ttc_noc_joint_count"
            ],
            'formula_cf_safe_rescue_count': formula_cf[
                "safe_rescue_count"
            ],
            'formula_cf_safe_progress_count': formula_cf[
                "safe_progress_count"
            ],
            'formula_cf_balanced_count': formula_cf["balanced_count"],
            'pred_ce_loss': pred_ce_loss,
            'pred_l1_loss': pred_l1_loss,
            'pred_area_loss': pred_area_loss,
            "inter_loss0": inter_loss0,
            # "inter_loss1": inter_loss1,
            "inter_loss": inter_loss,
            "min_loss0": min_loss0,
            # "min_loss1": min_loss1,
            "min_loss": min_loss,
            "score": score,
            "best_score": best_score,
            "future_loss": future_loss,
            "future_matched_distance": pred.get(
                "future_matched_distance",
                pred["pdm_score"].new_zeros(()).detach(),
            ).mean(),
            "ranking_loss": ranking_loss,
            "effective_ranking_weight": (
                formula_zero.detach().new_tensor(effective_ranking_weight)
            ),
            "scheduled_scorer_weight": formula_zero.detach().new_tensor(
                scheduled_scorer_weight
            ),
            "scheduled_future_weight": formula_zero.detach().new_tensor(
                scheduled_future_weight
            ),
            "effective_progress_ranking_weight": (
                formula_zero.detach().new_tensor(
                    effective_progress_ranking_weight
                )
            ),
            "progress_ranking_loss": progress_ranking_loss,
            "progress_ranking_pair_count": progress_ranking_pair_count,
            "progress_ranking_accuracy": progress_ranking_accuracy,
            "effective_safe_progress_tie_ranking_weight": (
                formula_zero.detach().new_tensor(
                    effective_safe_progress_tie_ranking_weight
                )
            ),
            "safe_progress_tie_ranking_loss": (
                safe_progress_tie_ranking_loss
            ),
            "safe_progress_tie_pair_count": safe_progress_tie_pair_count,
            "safe_progress_tie_ranking_accuracy": (
                safe_progress_tie_ranking_accuracy
            ),
            "formula_metric_loss": formula_losses["metric_loss"],
            "formula_noc_loss": formula_losses["noc_loss"],
            "formula_dac_loss": formula_losses["dac_loss"],
            "formula_ttc_loss": formula_losses["ttc_loss"],
            "formula_comfort_loss": formula_losses["comfort_loss"],
            "raw_progress_loss": formula_losses[
                "raw_progress_loss"
            ],
            "pdm_progress_loss": formula_losses[
                "pdm_progress_loss"
            ],
            "raw_progress_mae_m": formula_losses[
                "raw_progress_mae_m"
            ],
            "pdm_progress_mae_m": formula_losses[
                "pdm_progress_mae_m"
            ],
            "ep_mae": formula_losses["ep_mae"],
            "formula_mae": formula_losses["formula_mae"],
        }

        return loss_dict

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        return self.pad_loss(targets, pred, self._config, features=features)

    def get_optimizers(self):
        if (
            self._config.future_integration == "ema_jqtf"
            and self._config.ema_jqtf_scorer_head_mode
            == "formula_progress"
        ):
            generator_params = []
            scorer_params = []
            predictor_params = []
            for name, param in self._pad_model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("scorer."):
                    scorer_params.append(param)
                elif name.startswith("_future_predictor."):
                    predictor_params.append(param)
                else:
                    generator_params.append(param)

            parameter_groups = []
            if generator_params:
                parameter_groups.append(
                    {
                        "params": generator_params,
                        "lr": self._lr,
                        "name": "generator_representation",
                    }
                )
            if scorer_params:
                parameter_groups.append(
                    {
                        "params": scorer_params,
                        "lr": self._lr
                        * float(
                            self._config.ema_jqtf_formula_scorer_lr_scale
                        ),
                        "name": "formula_scorer",
                    }
                )
            if predictor_params:
                parameter_groups.append(
                    {
                        "params": predictor_params,
                        "lr": self._lr
                        * float(
                            self._config.ema_jqtf_formula_future_lr_scale
                        ),
                        "name": "future_predictor",
                    }
                )
            if not parameter_groups:
                raise RuntimeError(
                    "formula_progress optimizer found no trainable parameters"
                )

            optimizer = torch.optim.Adam(
                parameter_groups, lr=self._lr
            )
            milestone = int(
                self._config.ema_jqtf_formula_lr_milestone
            )
            gamma = float(
                self._config.ema_jqtf_formula_lr_gamma
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[milestone],
                gamma=gamma,
            )
            group_summary = ", ".join(
                f"{group['name']}={group['lr']:.2e}"
                f" ({len(group['params'])} tensors)"
                for group in parameter_groups
            )
            print(
                "Formula-progress optimizer groups: "
                f"{group_summary}; epoch {milestone} x{gamma:g}"
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        if self._config.freeze_encoder or self._config.use_lora:
            predictor_params = []
            other_params = []
            for name, param in self._pad_model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("_future_predictor"):
                    predictor_params.append(param)
                else:
                    other_params.append(param)
            if predictor_params and self._config.predict_future:
                return torch.optim.Adam([
                    {'params': other_params, 'lr': self._lr},
                    {'params': predictor_params, 'lr': self._lr * 5},
                ], lr=self._lr)
            return torch.optim.Adam([
                {'params': [p for p in self._pad_model.parameters() if p.requires_grad], 'lr': self._lr},
            ], lr=self._lr)
        # E2E: backbone with encoder_lr_scale, rest with full lr
        return torch.optim.Adam([
            {'params': self._pad_model._backbone.parameters(), 'lr': self._config.encoder_lr_scale * self._lr},
            {'params': [p for n, p in self._pad_model.named_parameters() if 'backbone' not in n], 'lr': self._lr},
        ], lr=self._lr)

    def update_future_target_ema(self) -> None:
        if not self._cache_data and hasattr(self, "_pad_model"):
            self._pad_model.update_future_target_ema()

    def get_training_callbacks(self):
        if (
            not self._cache_data
            and self._config.predict_future
            and self._config.future_target_mode == "ema_online_encoder"
            and not getattr(
                self._config, "scorer_only_training", False
            )
        ):
            return [FutureTargetEMACallback()]
        return []
