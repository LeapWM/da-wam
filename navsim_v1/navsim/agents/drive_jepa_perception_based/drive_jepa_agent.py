from typing import Any, List, Dict, NamedTuple, Union
import hashlib

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
    select_future_camera_features,
)
from navsim.agents.transfuser.transfuser_loss import _agent_loss
from .posttraj_future import build_current_future_pair
from .score_module.posttraj_stage2_sampler import (
    PostTrajStage2Sampler,
    Stage2SamplerConfig,
    anchor_score_cache_path,
    load_anchor_scores,
)
from .score_module.posttraj_final_ranking import (
    FinalTopKRankingOutput,
    final_topk_hard_ranking_loss,
)
from .score_module.posttraj_safety_hard import (
    SafetyHardOutput,
    safety_hard_loss,
    zero_safety_hard_output,
)
from .score_module.posttraj_safe_final_rank import (
    safe_final_ranking_loss,
    zero_safe_final_ranking_output,
)


class PostTrajAnchorLossOutput(NamedTuple):
    final_loss: torch.Tensor
    logits: torch.Tensor
    target_scores: torch.Tensor
    distance: torch.Tensor
    score_gap: torch.Tensor
    local_hard_count: torch.Tensor
    local_distance: torch.Tensor
    local_score_gap: torch.Tensor
    high_safe_count: torch.Tensor
    medium_count: torch.Tensor
    noc_failure_count: torch.Tensor
    dac_failure_count: torch.Tensor
    ttc_only_count: torch.Tensor


def posttraj_joint_weight(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    target_weight: float,
) -> float:
    """Warm up at zero, then linearly ramp to the target weight."""
    warmup = max(0, int(warmup_epochs))
    ramp = max(0, int(ramp_epochs))
    target = float(target_weight)
    if epoch < warmup:
        return 0.0
    if ramp == 0:
        return target
    progress = min(1.0, (epoch - warmup + 1) / float(ramp))
    return target * progress


def posttraj_joint_safety_scale(
    current_anchor_weight: float,
    target_anchor_weight: float,
) -> float:
    """Normalize the joint anchor schedule to a stable [0, 1] scale."""
    target = float(target_anchor_weight)
    if target <= 0.0:
        return 1.0
    return min(1.0, max(0.0, float(current_anchor_weight) / target))


def posttraj_stage2_schedule_value(
    values,
    epoch: int,
    fallback: float,
) -> float:
    """Read one epoch value, holding the last entry after the schedule ends."""
    if not values:
        return float(fallback)
    index = min(max(0, int(epoch)), len(values) - 1)
    return float(values[index])


class FutureTargetEMACallback(pl.Callback):
    """Update the target encoder exactly once after each optimizer step."""

    def __init__(self):
        super().__init__()
        self._last_global_step = 0

    def on_train_start(self, trainer, pl_module) -> None:
        self._last_global_step = int(trainer.global_step)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        global_step = int(trainer.global_step)
        if global_step <= self._last_global_step:
            return
        agent = getattr(pl_module, "agent", None)
        if agent is not None:
            agent.update_future_target_ema()
        self._last_global_step = global_step


class PostTrajJointWeightCallback(pl.Callback):
    """Advance the counterfactual weight once at each training epoch."""

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        agent = getattr(pl_module, "agent", None)
        if agent is not None:
            agent.set_posttraj_joint_epoch(int(trainer.current_epoch))


class PostTrajStage2ScheduleCallback(pl.Callback):
    """Advance Stage-2 loss weights and the shared optimizer LR by epoch."""

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        agent = getattr(pl_module, "agent", None)
        if agent is None:
            return
        learning_rate = agent.set_posttraj_stage2_epoch(
            int(trainer.current_epoch)
        )[-1]
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate


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
        self._checkpoint_path = checkpoint_path or ""
        self._cache_data = cache_data
        self._posttraj_joint_anchor_weight = 0.0
        self._posttraj_stage2_unsafe_final_weight = float(
            config.posttraj_safety_unsafe_final_weight
        )
        self._posttraj_stage2_safety_rank_weight = float(
            config.posttraj_safety_rank_weight
        )
        self._posttraj_stage2_safe_final_rank_weight = float(
            config.posttraj_safe_final_rank_weight
        )
        self._posttraj_stage2_lr = float(lr)
        self._posttraj_joint_unsafe_final_weight = float(
            config.posttraj_safety_unsafe_final_weight
        )
        self._posttraj_joint_safety_rank_weight = float(
            config.posttraj_safety_rank_weight
        )
        self._posttraj_joint_safe_final_rank_weight = float(
            config.posttraj_safe_final_rank_weight
        )

        if config.posttraj_stage2_enabled and config.posttraj_joint_enabled:
            raise ValueError(
                "posttraj_stage2_enabled and posttraj_joint_enabled are "
                "mutually exclusive"
            )

        if not self._cache_data:
            self._pad_model = DriveJEPAModel(config)

        training_setup = (
            not self._cache_data
            and (
                self._checkpoint_path == ""
                or config.posttraj_stage2_enabled
                or config.posttraj_joint_enabled
            )
        )
        if config.b2d:
            if config.b2d_expert_selection and (
                config.posttraj_safety_hard_enabled or config.posttraj_safety_rank_weight
                or config.posttraj_safe_final_rank_weight or config.posttraj_stage2_enabled
            ):
                raise ValueError('Expert selection must not mix rule-final ranking or Stage-2 objectives')
            from .score_module.b2d_rules_v2 import SUPPORTED_VERSIONS
            if config.b2d_scorer_version not in SUPPORTED_VERSIONS:
                raise ValueError('B2D requires an explicit supported rule label version')
            if config.agent_pred or config.area_pred:
                raise ValueError('B2D v2 uses six score targets; disable legacy agent_pred/area_pred heads')
            if config.posttraj_joint_enabled or (config.posttraj_stage2_enabled and config.posttraj_stage2_use_anchors):
                raise ValueError('NAVSIM anchor caches cannot be used for B2D v2')
            if config.command_num != 6:
                raise ValueError('B2D uses six commands and 13 ego-status features')
            from .score_module.b2d_v2_provider import B2DScoreProvider
            self._b2d_score_provider = B2DScoreProvider(config)
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
        if training_setup and not config.b2d:
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

            from .score_module.compute_navsim_score import get_scores

            metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/Drive-JEPA-cache/train_metric_cache"))
            self.train_metric_cache_paths = metric_cache.metric_cache_paths
            self.test_metric_cache_paths = metric_cache.metric_cache_paths

            self.get_scores = get_scores
        
        if not self._cache_data:
            anchor_path = config.anchor_trajectory_path or "./data/8192.npy"
            if config.b2d:
                self.anchors = np.empty((0, config.num_poses, 3), dtype=np.float32)
            else:
                poses = np.load(anchor_path)
                self.anchors = poses[:, 4::5]
            if (
                (
                    config.posttraj_stage2_enabled
                    and config.posttraj_stage2_use_anchors
                )
                or config.posttraj_joint_enabled
            ):
                exp_root = Path(os.environ["NAVSIM_EXP_ROOT"])
                self._anchor_metric_cache_root = (
                    exp_root / "Drive-JEPA-cache" / "train_metric_cache"
                )
                self._anchor_score_root = (
                    Path(config.posttraj_stage2_anchor_cache_root)
                    if config.posttraj_stage2_anchor_cache_root
                    else exp_root
                    / "Drive-JEPA-cache"
                    / "anchors_scores_v1_full_fp16"
                )
                self._stage2_sampler = PostTrajStage2Sampler(
                    self.anchors,
                    Stage2SamplerConfig(
                        num_local_hard=config.posttraj_stage2_num_local_hard,
                        num_balanced=config.posttraj_stage2_num_balanced,
                        nearest_per_proposal=(
                            config.posttraj_stage2_nearest_per_proposal
                        ),
                        min_score_gap=config.posttraj_stage2_min_score_gap,
                        yaw_weight=config.posttraj_stage2_yaw_weight,
                        sampling_profile=(
                            config.posttraj_stage2_sampling_profile
                        ),
                        seed=config.posttraj_stage2_seed,
                    ),
                )

        if not self._cache_data and config.posttraj_stage2_enabled:
            if not self._checkpoint_path:
                raise ValueError(
                    "PostTraj Stage 2 requires a Stage A checkpoint_path"
                )
            self.initialize()
            self._pad_model.configure_posttraj_stage2_trainable()
        elif (
            not self._cache_data
            and config.posttraj_joint_enabled
            and self._checkpoint_path
        ):
            self.initialize()

    def set_posttraj_joint_epoch(self, epoch: int) -> float:
        """Set and return the scheduled external-anchor weight."""
        if not self._config.posttraj_joint_enabled:
            self._posttraj_joint_anchor_weight = 0.0
            return 0.0
        weight = posttraj_joint_weight(
            epoch,
            self._config.posttraj_joint_warmup_epochs,
            self._config.posttraj_joint_ramp_epochs,
            self._config.posttraj_joint_anchor_loss_weight,
        )
        self._posttraj_joint_anchor_weight = weight
        if self._config.posttraj_joint_safety_schedule_enabled:
            self._posttraj_joint_unsafe_final_weight = (
                posttraj_stage2_schedule_value(
                    self._config
                    .posttraj_joint_unsafe_final_weight_schedule,
                    epoch,
                    self._config.posttraj_safety_unsafe_final_weight,
                )
            )
            self._posttraj_joint_safety_rank_weight = (
                posttraj_stage2_schedule_value(
                    self._config
                    .posttraj_joint_safety_rank_weight_schedule,
                    epoch,
                    self._config.posttraj_safety_rank_weight,
                )
            )
            self._posttraj_joint_safe_final_rank_weight = (
                posttraj_stage2_schedule_value(
                    self._config
                    .posttraj_joint_safe_final_rank_weight_schedule,
                    epoch,
                    self._config.posttraj_safe_final_rank_weight,
                )
            )
        return weight

    def set_posttraj_stage2_epoch(self, epoch: int):
        """Set Stage-2-only scheduled weights and return all four values."""
        if not (
            self._config.posttraj_stage2_enabled
            and self._config.posttraj_stage2_schedule_enabled
        ):
            return (
                self._posttraj_stage2_unsafe_final_weight,
                self._posttraj_stage2_safety_rank_weight,
                self._posttraj_stage2_safe_final_rank_weight,
                self._posttraj_stage2_lr,
            )
        self._posttraj_stage2_unsafe_final_weight = (
            posttraj_stage2_schedule_value(
                self._config
                .posttraj_stage2_unsafe_final_weight_schedule,
                epoch,
                self._config.posttraj_safety_unsafe_final_weight,
            )
        )
        self._posttraj_stage2_safety_rank_weight = (
            posttraj_stage2_schedule_value(
                self._config.posttraj_stage2_safety_rank_weight_schedule,
                epoch,
                self._config.posttraj_safety_rank_weight,
            )
        )
        self._posttraj_stage2_safe_final_rank_weight = (
            posttraj_stage2_schedule_value(
                self._config
                .posttraj_stage2_safe_final_rank_weight_schedule,
                epoch,
                self._config.posttraj_safe_final_rank_weight,
            )
        )
        self._posttraj_stage2_lr = posttraj_stage2_schedule_value(
            self._config.posttraj_stage2_lr_schedule,
            epoch,
            self._lr,
        )
        return (
            self._posttraj_stage2_unsafe_final_weight,
            self._posttraj_stage2_safety_rank_weight,
            self._posttraj_stage2_safe_final_rank_weight,
            self._posttraj_stage2_lr,
        )

    def name(self) -> str:
        """Inherited, see superclass."""
        return 'drive_jepa_perception_based_agent' 

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path != "":
            state_dict: Dict[str, Any] = torch.load(
                self._checkpoint_path, map_location="cpu"
            )["state_dict"]
            mapped = {
                key.replace("agent._pad_model", "_pad_model"): value
                for key, value in state_dict.items()
            }
            self.load_state_dict(mapped, strict=True)

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
        if self._config.b2d:
            detached = proposals.detach()
            results = self._b2d_score_provider.score(
                targets['token'], detached.float().cpu().numpy(), self.training
            )
            labels = torch.as_tensor(np.stack([r['scores'] for r in results]), device=proposals.device)
            if self._config.b2d_expert_selection:
                from .score_module.b2d_expert_objective import expert_quality
                labels[..., -1] = expert_quality(
                    detached, targets['trajectory'], self._config.b2d_expert_scale_m,
                    self._config.b2d_expert_yaw_weight_m)
            final = labels[..., -1]
            if test:
                l2 = torch.linalg.norm(detached[:, 0] - targets['trajectory'], dim=-1)[:, :4]
                return final[:, 0].mean(), final.amax(-1).mean(), final, l2.mean(), labels[:, 0]
            b, p, t = proposals.shape[:3]
            # Disabled legacy auxiliary heads do not consume these placeholders.
            states = proposals.new_zeros((b, p, t, 2, 4, 2))
            valid = torch.zeros((b, p, t, 2), dtype=torch.bool, device=proposals.device)
            areas = torch.as_tensor(np.stack([r['road_by_time'] for r in results]), device=proposals.device)
            areas = torch.stack([areas, areas], dim=-1)
            indices = [np.empty(0, dtype=np.int64) for _ in results]
            return final, final.amax(-1), labels, states, valid, areas, indices
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        target_trajectory = targets["trajectory"]
        proposals=proposals.detach()

        data_points = [
            {
                "token": metric_cache_paths[token],
                "poses": poses,
                "test": test
            }
            for token, poses in zip(targets["token"], proposals.float().cpu().numpy())
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

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas, scores_index

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

        sub_score_loss = self.bce_logit_loss(pred_logit, target_scores[..., -pred_logit.shape[-1]:])  # .mean()[..., -6:]
        if self._config.b2d and self._config.b2d_expert_selection:
            sub_score_loss = self.bce_logit_loss(pred_logit[..., :5], target_scores[..., :5])

        final_score_loss = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])  # .mean()

        if pred_logit2 is not None:
            sub_score_loss2 = self.bce_logit_loss(pred_logit2, target_scores)  # .mean()[..., -6:-1][..., -6:-1]
            if self._config.b2d and self._config.b2d_expert_selection:
                sub_score_loss2 = self.bce_logit_loss(pred_logit2[..., :5], target_scores[..., :5])

            final_score_loss2 = self.bce_logit_loss(pred_logit2[..., -1], target_scores[..., -1])  # .mean()

            sub_score_loss=(sub_score_loss+sub_score_loss2)/2

            final_score_loss=(final_score_loss+final_score_loss2)/2

        return sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss

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
        layers = proposal_list if config.b2d else proposal_list[:len(scores_index)]
        for proposals_i in layers:
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

    def _stage2_scene_seed(self, token: Any) -> int:
        digest = hashlib.sha256(str(token).encode("utf-8")).digest()
        return self._config.posttraj_stage2_seed ^ int.from_bytes(
            digest[:4], "little"
        )

    def posttraj_anchor_loss(
        self,
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
        target_scores: torch.Tensor,
    ):
        """Score true external anchors through trajectory-conditioned future."""
        proposals_np = pred["proposals"].detach().float().cpu().numpy()
        generated_final_np = (
            target_scores[..., 5].detach().float().cpu().numpy()
        )
        metric_paths = (
            self.train_metric_cache_paths
            if self.training
            else self.test_metric_cache_paths
        )
        sampled_trajectories = []
        sampled_score_targets = []
        sampled_distances = []
        sampled_gaps = []
        sampled_local_counts = []
        sampled_local_distances = []
        sampled_local_gaps = []
        sampled_high_safe_counts = []
        sampled_medium_counts = []
        sampled_noc_failure_counts = []
        sampled_dac_failure_counts = []
        sampled_ttc_only_counts = []
        for batch_index, token in enumerate(targets["token"]):
            score_path = anchor_score_cache_path(
                Path(metric_paths[token]),
                self._anchor_metric_cache_root,
                self._anchor_score_root,
            )
            anchor_scores = load_anchor_scores(score_path)
            sample = self._stage2_sampler.sample(
                proposals_np[batch_index],
                generated_final_np[batch_index],
                anchor_scores,
                seed=self._stage2_scene_seed(token),
            )
            sampled_trajectories.append(self.anchors[sample.indices])
            sampled_score_targets.append(anchor_scores[sample.indices])
            sampled_distances.append(sample.distances)
            sampled_gaps.append(sample.score_gaps)
            sampled_local_counts.append(sample.local_hard_count)
            sampled_local_distances.append(
                sample.distances[sample.is_local_hard]
            )
            sampled_local_gaps.append(
                sample.score_gaps[sample.is_local_hard]
            )
            sampled_high_safe_counts.append(
                int(np.count_nonzero(sample.buckets == "high_safe"))
            )
            sampled_medium_counts.append(
                int(np.count_nonzero(sample.buckets == "medium"))
            )
            sampled_noc_failure_counts.append(
                int(np.count_nonzero(sample.buckets == "noc_failure"))
            )
            sampled_dac_failure_counts.append(
                int(np.count_nonzero(sample.buckets == "dac_failure"))
            )
            sampled_ttc_only_counts.append(
                int(np.count_nonzero(sample.buckets == "ttc_only"))
            )

        scene_map = pred["future_scene_map"]
        anchors = torch.as_tensor(
            np.stack(sampled_trajectories),
            device=scene_map.device,
            dtype=scene_map.dtype,
        )
        anchor_targets = torch.as_tensor(
            np.stack(sampled_score_targets),
            device=scene_map.device,
            dtype=torch.float32,
        )
        anchor_logits = self._pad_model.score_external_trajectories(
            anchors, scene_map
        )
        anchor_final_loss = F.binary_cross_entropy_with_logits(
            anchor_logits[..., 5].float(), anchor_targets[..., 5]
        )

        def metric(values):
            array = np.asarray(values, dtype=np.float32)
            if array.size == 0:
                return anchor_final_loss.detach() * 0.0
            return torch.as_tensor(array, device=scene_map.device).mean()

        return PostTrajAnchorLossOutput(
            final_loss=anchor_final_loss,
            logits=anchor_logits,
            target_scores=anchor_targets,
            distance=metric(np.concatenate(sampled_distances)),
            score_gap=metric(np.concatenate(sampled_gaps)),
            local_hard_count=metric(sampled_local_counts),
            local_distance=metric(np.concatenate(sampled_local_distances)),
            local_score_gap=metric(np.concatenate(sampled_local_gaps)),
            high_safe_count=metric(sampled_high_safe_counts),
            medium_count=metric(sampled_medium_counts),
            noc_failure_count=metric(sampled_noc_failure_counts),
            dac_failure_count=metric(sampled_dac_failure_counts),
            ttc_only_count=metric(sampled_ttc_only_counts),
        )

    def posttraj_stage2_loss(
        self,
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
        final_scores: torch.Tensor,
        best_scores: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Frozen-generator scorer calibration with optional anchors/rank."""
        generated_final_loss = F.binary_cross_entropy_with_logits(
            pred["pred_logit"][..., 5].float(),
            target_scores[..., 5].float(),
        )
        zero = generated_final_loss * 0.0
        unsafe_final_weight = float(getattr(
            self,
            "_posttraj_stage2_unsafe_final_weight",
            self._config.posttraj_safety_unsafe_final_weight,
        ))
        safety_rank_weight = float(getattr(
            self,
            "_posttraj_stage2_safety_rank_weight",
            self._config.posttraj_safety_rank_weight,
        ))
        safe_final_rank_weight = float(getattr(
            self,
            "_posttraj_stage2_safe_final_rank_weight",
            self._config.posttraj_safe_final_rank_weight,
        ))
        stage2_lr = float(getattr(
            self,
            "_posttraj_stage2_lr",
            self._lr if hasattr(self, "_lr") else 0.0,
        ))
        anchor_output = None
        if self._config.posttraj_stage2_use_anchors:
            anchor_output = self.posttraj_anchor_loss(
                targets, pred, target_scores
            )
            anchor_final_loss = anchor_output.final_loss
            anchor_distance = anchor_output.distance
            anchor_score_gap = anchor_output.score_gap
            anchor_local_hard_count = anchor_output.local_hard_count
            anchor_local_distance = anchor_output.local_distance
            anchor_local_score_gap = anchor_output.local_score_gap
            anchor_high_safe_count = anchor_output.high_safe_count
            anchor_medium_count = anchor_output.medium_count
            anchor_noc_failure_count = anchor_output.noc_failure_count
            anchor_dac_failure_count = anchor_output.dac_failure_count
            anchor_ttc_only_count = anchor_output.ttc_only_count
        else:
            anchor_final_loss = zero
            anchor_distance = zero.detach()
            anchor_score_gap = zero.detach()
            anchor_local_hard_count = zero.detach()
            anchor_local_distance = zero.detach()
            anchor_local_score_gap = zero.detach()
            anchor_high_safe_count = zero.detach()
            anchor_medium_count = zero.detach()
            anchor_noc_failure_count = zero.detach()
            anchor_dac_failure_count = zero.detach()
            anchor_ttc_only_count = zero.detach()

        rank_weight = float(
            self._config.posttraj_final_topk_rank_weight
        )
        if rank_weight > 0.0:
            rank_output = final_topk_hard_ranking_loss(
                torch.sigmoid(pred["pred_logit"][..., 5].float()),
                target_scores,
                topk=self._config.posttraj_final_topk_rank_k,
                score_gap=(
                    self._config.posttraj_final_topk_rank_score_gap
                ),
                margin_cap=(
                    self._config.posttraj_final_topk_rank_margin_cap
                ),
                false_topk_weight=(
                    self._config
                    .posttraj_final_topk_rank_false_topk_weight
                ),
                safety_pair_weight=(
                    self._config.posttraj_final_topk_rank_safety_weight
                ),
                safety_threshold=(
                    self._config
                    .posttraj_final_topk_rank_safety_threshold
                ),
            )
        else:
            rank_output = FinalTopKRankingOutput(
                loss=zero,
                pair_count=zero.detach(),
                accuracy=zero.detach(),
                topk_overlap=zero.detach(),
                true_best_in_pred_topk=zero.detach(),
                unsafe_in_pred_topk=zero.detach(),
                safety_pair_count=zero.detach(),
                safety_pair_accuracy=zero.detach(),
            )
        if self._config.posttraj_safety_hard_enabled:
            generated_safety = safety_hard_loss(
                pred["pred_logit"],
                target_scores,
                safety_threshold=self._config.posttraj_safety_threshold,
                unsafe_final_weight=unsafe_final_weight,
                topk=self._config.posttraj_safety_rank_topk,
                margin=self._config.posttraj_safety_rank_margin,
            )
            if anchor_output is not None:
                anchor_safety = safety_hard_loss(
                    anchor_output.logits,
                    anchor_output.target_scores,
                    safety_threshold=(
                        self._config.posttraj_safety_threshold
                    ),
                    unsafe_final_weight=unsafe_final_weight,
                    topk=self._config.posttraj_safety_rank_topk,
                    margin=self._config.posttraj_safety_rank_margin,
                )
            else:
                anchor_safety = zero_safety_hard_output(zero)
        else:
            generated_safety = zero_safety_hard_output(zero)
            anchor_safety = zero_safety_hard_output(zero)

        anchor_weight = float(
            self._config.posttraj_stage2_anchor_loss_weight
        )
        if safe_final_rank_weight > 0.0:
            generated_safe_rank = safe_final_ranking_loss(
                pred["pred_logit"],
                target_scores,
                safety_threshold=self._config.posttraj_safety_threshold,
                topk=self._config.posttraj_safe_final_rank_topk,
                score_gap=self._config.posttraj_safe_final_rank_score_gap,
                margin_cap=(
                    self._config.posttraj_safe_final_rank_margin_cap
                ),
            )
            if anchor_output is not None:
                anchor_safe_rank = safe_final_ranking_loss(
                    anchor_output.logits,
                    anchor_output.target_scores,
                    safety_threshold=(
                        self._config.posttraj_safety_threshold
                    ),
                    topk=self._config.posttraj_safe_final_rank_topk,
                    score_gap=(
                        self._config.posttraj_safe_final_rank_score_gap
                    ),
                    margin_cap=(
                        self._config.posttraj_safe_final_rank_margin_cap
                    ),
                )
            else:
                anchor_safe_rank = zero_safe_final_ranking_output(zero)
        else:
            generated_safe_rank = zero_safe_final_ranking_output(zero)
            anchor_safe_rank = zero_safe_final_ranking_output(zero)
        safe_final_rank_loss = (
            generated_safe_rank.loss
            + anchor_weight * anchor_safe_rank.loss
        )
        if self._config.posttraj_safety_hard_enabled:
            generated_final_objective = generated_safety.weighted_final_loss
            anchor_final_objective = anchor_safety.weighted_final_loss
        else:
            generated_final_objective = generated_final_loss
            anchor_final_objective = anchor_final_loss
        safety_weighted_final_loss = (
            generated_final_objective
            + anchor_weight * anchor_final_objective
        )
        safety_rank_loss = (
            generated_safety.rank_loss
            + anchor_weight * anchor_safety.rank_loss
        )
        loss = (
            self._config.final_score_weight * generated_final_objective
            + anchor_weight * anchor_final_objective
            + rank_weight * rank_output.loss
            + safety_rank_weight * safety_rank_loss
            + safe_final_rank_weight * safe_final_rank_loss
        )
        pdm_score = pred["pdm_score"].detach()
        selected = torch.argmax(pdm_score, dim=1)
        batch_index = torch.arange(
            len(final_scores), device=final_scores.device
        )
        score = final_scores[batch_index, selected].mean()
        metric_zero = generated_final_loss.detach() * 0.0
        return {
            "loss": loss,
            "trajectory_loss": metric_zero,
            "sub_score_loss": metric_zero,
            "final_score_loss": generated_final_objective,
            "generated_final_loss": generated_final_objective,
            "generated_final_bce": generated_final_loss,
            "anchor_final_loss": anchor_final_objective,
            "anchor_final_bce": anchor_final_loss,
            "anchor_distance": anchor_distance,
            "anchor_score_gap": anchor_score_gap,
            "anchor_local_hard_count": anchor_local_hard_count,
            "anchor_local_distance": anchor_local_distance,
            "anchor_local_score_gap": anchor_local_score_gap,
            "anchor_high_safe_count": anchor_high_safe_count,
            "anchor_medium_count": anchor_medium_count,
            "anchor_noc_failure_count": anchor_noc_failure_count,
            "anchor_dac_failure_count": anchor_dac_failure_count,
            "anchor_ttc_only_count": anchor_ttc_only_count,
            "final_topk_rank_weight": metric_zero.new_tensor(rank_weight),
            "final_topk_rank_loss": rank_output.loss,
            "final_topk_rank_pair_count": rank_output.pair_count,
            "final_topk_rank_accuracy": rank_output.accuracy,
            "final_topk_overlap": rank_output.topk_overlap,
            "true_best_in_pred_topk": rank_output.true_best_in_pred_topk,
            "unsafe_in_pred_topk": rank_output.unsafe_in_pred_topk,
            "safety_rank_pair_count": rank_output.safety_pair_count,
            "safety_rank_accuracy": rank_output.safety_pair_accuracy,
            "safety_hard_weighted_final_loss": safety_weighted_final_loss,
            "safety_hard_rank_loss": safety_rank_loss,
            "stage2_unsafe_final_weight": metric_zero.new_tensor(
                unsafe_final_weight
            ),
            "stage2_safety_rank_weight": metric_zero.new_tensor(
                safety_rank_weight
            ),
            "stage2_safe_final_rank_weight": metric_zero.new_tensor(
                safe_final_rank_weight
            ),
            "stage2_lr": metric_zero.new_tensor(stage2_lr),
            "safe_final_rank_loss": safe_final_rank_loss,
            "generated_safe_final_rank_pair_count": (
                generated_safe_rank.pair_count
            ),
            "generated_safe_final_rank_accuracy": (
                generated_safe_rank.accuracy
            ),
            "anchor_safe_final_rank_pair_count": (
                anchor_safe_rank.pair_count
            ),
            "anchor_safe_final_rank_accuracy": anchor_safe_rank.accuracy,
            "generated_safety_rank_pair_count": generated_safety.pair_count,
            "generated_safety_rank_accuracy": generated_safety.rank_accuracy,
            "generated_unsafe_fraction": generated_safety.unsafe_fraction,
            "generated_unsafe_in_pred_topk": (
                generated_safety.unsafe_in_pred_topk
            ),
            "generated_selected_unsafe_rate": (
                generated_safety.selected_unsafe_rate
            ),
            "generated_safe_available_rate": (
                generated_safety.safe_available_rate
            ),
            "generated_unsafe_overestimate_rate": (
                generated_safety.unsafe_overestimate_rate
            ),
            "generated_noc_failure_overestimate_rate": (
                generated_safety.noc_failure_overestimate_rate
            ),
            "generated_dac_failure_overestimate_rate": (
                generated_safety.dac_failure_overestimate_rate
            ),
            "generated_ttc_failure_overestimate_rate": (
                generated_safety.ttc_failure_overestimate_rate
            ),
            "anchor_safety_rank_pair_count": anchor_safety.pair_count,
            "anchor_safety_rank_accuracy": anchor_safety.rank_accuracy,
            "anchor_unsafe_fraction": anchor_safety.unsafe_fraction,
            "anchor_unsafe_in_pred_topk": anchor_safety.unsafe_in_pred_topk,
            "anchor_selected_unsafe_rate": (
                anchor_safety.selected_unsafe_rate
            ),
            "anchor_safe_available_rate": anchor_safety.safe_available_rate,
            "anchor_unsafe_overestimate_rate": (
                anchor_safety.unsafe_overestimate_rate
            ),
            "anchor_noc_failure_overestimate_rate": (
                anchor_safety.noc_failure_overestimate_rate
            ),
            "anchor_dac_failure_overestimate_rate": (
                anchor_safety.dac_failure_overestimate_rate
            ),
            "anchor_ttc_failure_overestimate_rate": (
                anchor_safety.ttc_failure_overestimate_rate
            ),
            "pred_ce_loss": metric_zero,
            "pred_l1_loss": metric_zero,
            "pred_area_loss": metric_zero,
            "inter_loss0": metric_zero,
            "inter_loss": metric_zero,
            "min_loss0": metric_zero,
            "min_loss": metric_zero,
            "future_loss": metric_zero,
            "future_matched_distance": metric_zero,
            "score": score,
            "best_score": best_scores.mean(),
        }

    def pad_loss(self, targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config  ):

        proposals = pred["proposals"]
        proposal_list = pred["proposal_list"]
        target_trajectory = targets["trajectory"]

        final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas, scores_index = self.compute_score(
            targets, proposals, test=False)

        if config.posttraj_stage2_enabled:
            return self.posttraj_stage2_loss(
                targets, pred, final_scores, best_scores, target_scores
            )
        
        trajectory_loss, min_loss, inter_loss, min_loss_list, inter_loss_list = self.trajectory_loss_anchors(proposal_list, target_trajectory, config, scores_index)

        min_loss0 = min_loss_list[0]
        inter_loss0 = inter_loss_list[0]
        # min_loss1 = min_loss_list[1]
        # inter_loss1 = inter_loss_list[1]

        if "pred_logit" in pred.keys():
            sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
                pred["pred_logit"],pred["pred_logit2"],
                pred["pred_agents_states"], pred["pred_area_logit"]
                , target_scores, gt_states, gt_valid, gt_ego_areas)
        else:
            sub_score_loss = final_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0

        safety_reference = final_score_loss
        if not torch.is_tensor(safety_reference):
            safety_reference = proposals.sum() * 0.0
        joint_unsafe_final_weight = float(getattr(
            self,
            "_posttraj_joint_unsafe_final_weight",
            config.posttraj_safety_unsafe_final_weight,
        ))
        joint_safety_rank_weight = float(getattr(
            self,
            "_posttraj_joint_safety_rank_weight",
            config.posttraj_safety_rank_weight,
        ))
        joint_safe_final_rank_weight = float(getattr(
            self,
            "_posttraj_joint_safe_final_rank_weight",
            config.posttraj_safe_final_rank_weight,
        ))
        generated_safety = zero_safety_hard_output(safety_reference)
        generated_safe_rank = zero_safe_final_ranking_output(
            safety_reference
        )
        joint_safety_scale = safety_reference.detach() * 0.0
        generated_final_objective = final_score_loss
        if config.posttraj_safety_hard_enabled:
            if "pred_logit" not in pred:
                raise RuntimeError(
                    "final-only safety training requires pred_logit"
                )
            generated_safety = safety_hard_loss(
                pred["pred_logit"],
                target_scores,
                safety_threshold=config.posttraj_safety_threshold,
                unsafe_final_weight=joint_unsafe_final_weight,
                topk=config.posttraj_safety_rank_topk,
                margin=config.posttraj_safety_rank_margin,
            )
            if config.posttraj_joint_enabled:
                scale = posttraj_joint_safety_scale(
                    self._posttraj_joint_anchor_weight,
                    config.posttraj_joint_anchor_loss_weight,
                )
            else:
                scale = 1.0
            joint_safety_scale = safety_reference.new_tensor(scale)
            generated_final_objective = (
                final_score_loss
                + joint_safety_scale
                * (
                    generated_safety.weighted_final_loss
                    - final_score_loss
                )
            )
        if joint_safe_final_rank_weight > 0.0:
            if "pred_logit" not in pred:
                raise RuntimeError(
                    "safe final ranking requires pred_logit"
                )
            generated_safe_rank = safe_final_ranking_loss(
                pred["pred_logit"],
                target_scores,
                safety_threshold=config.posttraj_safety_threshold,
                topk=config.posttraj_safe_final_rank_topk,
                score_gap=config.posttraj_safe_final_rank_score_gap,
                margin_cap=config.posttraj_safe_final_rank_margin_cap,
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

        loss = (
                config.trajectory_weight * trajectory_loss
                + config.sub_score_weight * sub_score_loss
                + config.final_score_weight * generated_final_objective
                + config.pred_ce_weight * pred_ce_loss
                + config.pred_l1_weight * pred_l1_loss
                + config.pred_area_weight * pred_area_loss
                + config.agent_class_weight * agent_class_loss
                + config.agent_box_weight * agent_box_loss
                + config.bev_semantic_weight * bev_semantic_loss

        )

        future_loss = loss.new_zeros(())
        future_matched_distance = loss.new_zeros(())
        horizon_losses = []
        if config.posttraj_future_enabled:
            if "future_camera_features" not in targets:
                raise RuntimeError(
                    "posttraj future training requires cached "
                    "future_camera_features"
                )
            requested_offsets = tuple(
                int(value)
                for value in config.posttraj_future_frame_offsets
            )
            future_cameras = select_future_camera_features(targets, requested_offsets)
            from .posttraj_future.multihorizon import multihorizon_loss
            future_loss, future_matched_index, matched_distance, horizon_losses = multihorizon_loss(
                self._pad_model, pred, proposals, target_trajectory,
                future_cameras, requested_offsets, config.posttraj_future_layernorm_target)
            future_matched_distance = matched_distance.detach().mean()
            pred["future_matched_index"] = future_matched_index
            loss = loss + config.posttraj_future_loss_weight * future_loss

        anchor_final_loss = loss.new_zeros(())
        anchor_distance = loss.new_zeros(())
        anchor_score_gap = loss.new_zeros(())
        anchor_local_hard_count = loss.new_zeros(())
        anchor_local_distance = loss.new_zeros(())
        anchor_local_score_gap = loss.new_zeros(())
        anchor_high_safe_count = loss.new_zeros(())
        anchor_medium_count = loss.new_zeros(())
        anchor_noc_failure_count = loss.new_zeros(())
        anchor_dac_failure_count = loss.new_zeros(())
        anchor_ttc_only_count = loss.new_zeros(())
        anchor_final_bce = loss.new_zeros(())
        anchor_safety = zero_safety_hard_output(loss.new_zeros(()))
        anchor_safe_rank = zero_safe_final_ranking_output(
            loss.new_zeros(())
        )
        joint_anchor_weight = loss.new_tensor(
            self._posttraj_joint_anchor_weight
        )
        if config.posttraj_joint_enabled and self._posttraj_joint_anchor_weight > 0:
            anchor_output = self.posttraj_anchor_loss(
                targets, pred, target_scores
            )
            anchor_final_loss = anchor_output.final_loss
            anchor_distance = anchor_output.distance
            anchor_score_gap = anchor_output.score_gap
            anchor_local_hard_count = anchor_output.local_hard_count
            anchor_local_distance = anchor_output.local_distance
            anchor_local_score_gap = anchor_output.local_score_gap
            anchor_high_safe_count = anchor_output.high_safe_count
            anchor_medium_count = anchor_output.medium_count
            anchor_noc_failure_count = anchor_output.noc_failure_count
            anchor_dac_failure_count = anchor_output.dac_failure_count
            anchor_ttc_only_count = anchor_output.ttc_only_count
            anchor_final_bce = anchor_final_loss
            if config.posttraj_safety_hard_enabled:
                anchor_safety = safety_hard_loss(
                    anchor_output.logits,
                    anchor_output.target_scores,
                    safety_threshold=config.posttraj_safety_threshold,
                    unsafe_final_weight=joint_unsafe_final_weight,
                    topk=config.posttraj_safety_rank_topk,
                    margin=config.posttraj_safety_rank_margin,
                )
                anchor_final_loss = anchor_safety.weighted_final_loss
            if joint_safe_final_rank_weight > 0.0:
                anchor_safe_rank = safe_final_ranking_loss(
                    anchor_output.logits,
                    anchor_output.target_scores,
                    safety_threshold=config.posttraj_safety_threshold,
                    topk=config.posttraj_safe_final_rank_topk,
                    score_gap=config.posttraj_safe_final_rank_score_gap,
                    margin_cap=config.posttraj_safe_final_rank_margin_cap,
                )
            loss = loss + joint_anchor_weight * anchor_final_loss

        safety_rank_loss = (
            joint_safety_scale * generated_safety.rank_loss
            + joint_anchor_weight * anchor_safety.rank_loss
        )
        safe_final_rank_loss = (
            joint_safety_scale * generated_safe_rank.loss
            + joint_anchor_weight * anchor_safe_rank.loss
        )
        loss = (
            loss
            + joint_safety_rank_weight * safety_rank_loss
            + joint_safe_final_rank_weight * safe_final_rank_loss
        )

        pdm_score = pred["pdm_score"].detach()
        top_proposals = torch.argmax(pdm_score, dim=1)
        score = final_scores[np.arange(len(final_scores)), top_proposals].mean()
        best_score = best_scores.mean()

        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            'sub_score_loss': sub_score_loss,
            "sub_score_loss_contribution": (
                config.sub_score_weight * sub_score_loss
            ),
            'final_score_loss': generated_final_objective,
            "generated_final_bce": final_score_loss,
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
            **{f"future_loss_h{j+1}": value for j, value in enumerate(horizon_losses)},
            "future_matched_distance": future_matched_distance,
            "joint_anchor_weight": joint_anchor_weight,
            "joint_safety_scale": joint_safety_scale,
            "joint_unsafe_final_weight": loss.new_tensor(
                joint_unsafe_final_weight
            ),
            "joint_safety_rank_weight": loss.new_tensor(
                joint_safety_rank_weight
            ),
            "joint_safe_final_rank_weight": loss.new_tensor(
                joint_safe_final_rank_weight
            ),
            "anchor_final_loss": anchor_final_loss,
            "anchor_final_bce": anchor_final_bce,
            "anchor_distance": anchor_distance,
            "anchor_score_gap": anchor_score_gap,
            "anchor_local_hard_count": anchor_local_hard_count,
            "anchor_local_distance": anchor_local_distance,
            "anchor_local_score_gap": anchor_local_score_gap,
            "anchor_high_safe_count": anchor_high_safe_count,
            "anchor_medium_count": anchor_medium_count,
            "anchor_noc_failure_count": anchor_noc_failure_count,
            "anchor_dac_failure_count": anchor_dac_failure_count,
            "anchor_ttc_only_count": anchor_ttc_only_count,
            "safety_hard_rank_loss": safety_rank_loss,
            "safe_final_rank_loss": safe_final_rank_loss,
            "generated_safe_final_rank_pair_count": (
                generated_safe_rank.pair_count
            ),
            "generated_safe_final_rank_accuracy": (
                generated_safe_rank.accuracy
            ),
            "anchor_safe_final_rank_pair_count": (
                anchor_safe_rank.pair_count
            ),
            "anchor_safe_final_rank_accuracy": anchor_safe_rank.accuracy,
            "generated_safety_rank_pair_count": generated_safety.pair_count,
            "generated_safety_rank_accuracy": generated_safety.rank_accuracy,
            "generated_unsafe_fraction": generated_safety.unsafe_fraction,
            "generated_unsafe_in_pred_topk": (
                generated_safety.unsafe_in_pred_topk
            ),
            "generated_selected_unsafe_rate": (
                generated_safety.selected_unsafe_rate
            ),
            "anchor_safety_rank_pair_count": anchor_safety.pair_count,
            "anchor_safety_rank_accuracy": anchor_safety.rank_accuracy,
            "anchor_unsafe_fraction": anchor_safety.unsafe_fraction,
            "anchor_unsafe_in_pred_topk": anchor_safety.unsafe_in_pred_topk,
            "anchor_selected_unsafe_rate": (
                anchor_safety.selected_unsafe_rate
            ),
        }

        if config.b2d and config.b2d_expert_selection:
            from .score_module.b2d_expert_objective import matching_classification_loss
            matching_ce = matching_classification_loss(pred['pred_logit'][..., -1], final_scores)
            if pred['pred_logit2'] is not None:
                matching_ce = .5 * (matching_ce + matching_classification_loss(pred['pred_logit2'][..., -1], final_scores))
            loss_dict['matching_ce_loss'] = matching_ce
            loss_dict['loss'] = loss_dict['loss'] + config.b2d_matching_ce_weight * matching_ce
        return loss_dict

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        return self.pad_loss(targets, pred, self._config)

    def get_optimizers(self):
        if self._config.freeze_encoder or self._config.use_lora:
            return torch.optim.Adam([
                {'params': [p for p in self._pad_model.parameters() if p.requires_grad], 'lr': self._lr},
            ], lr=self._lr)
        # E2E: backbone with encoder_lr_scale, rest with full lr
        return torch.optim.Adam([
            {'params': self._pad_model._backbone.parameters(), 'lr': self._config.encoder_lr_scale * self._lr},
            {'params': [p for n, p in self._pad_model.named_parameters() if 'backbone' not in n], 'lr': self._lr},
        ], lr=self._lr)

    def update_future_target_ema(self) -> None:
        if not self._cache_data:
            self._pad_model.update_future_target_ema()

    def get_training_callbacks(self):
        callbacks = []
        if (
            self._config.posttraj_future_enabled
            and not self._config.posttraj_stage2_enabled
        ):
            callbacks.append(FutureTargetEMACallback())
        if self._config.posttraj_joint_enabled:
            callbacks.append(PostTrajJointWeightCallback())
        if (
            self._config.posttraj_stage2_enabled
            and self._config.posttraj_stage2_schedule_enabled
        ):
            callbacks.append(PostTrajStage2ScheduleCallback())
        return callbacks
