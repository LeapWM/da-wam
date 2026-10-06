from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class DriveJEPAConfig:
    b2d: bool = False
    # Offline supervision generated from the compressed Bench2Drive clips.
    # These are read only when b2d=True and the agent is constructed for
    # training/validation, so zero-shot CARLA inference remains unchanged.
    b2d_train_scorer_cache_path: str = ""
    b2d_val_scorer_cache_path: str = ""
    b2d_map_cache_path: str = ""

    ref_num: int=4

    traj_bev: bool=True
    traj_proposal_query: bool=True

    double_score: bool=False

    agent_pred: bool=True
    area_pred: bool=True

    bev_map: bool=False
    bev_agent: bool=False

    proposal_num: int = 32
    point_cloud_range = [-32, -32, -2.0, 32, 32, 6.0]
    num_points_in_pillar: int=4

    half_length: float= 2.588 +0.25 #small buffer for safety
    half_width: float =1.1485 +0.1
    rear_axle_to_center: float = 1.461
    lidar_height: float = 0

    num_poses: int=8
    command_num: int=4

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0
    num_bev_layers: int=1
    image_architecture: str = "vit_large"

    # V-JEPA backbone
    vjepa_version: str = "2.0"
    pretrain_pt_path: str = "/mnt/downloads-1/models/vjepa2/vitl.pt"
    freeze_encoder: bool = False
    use_lora: bool = False
    lora_rank: int = 32
    encoder_lr_scale: float = 0.1

    # loss weights
    trajectory_weight: float = 1
    inter_weight: float = 0
    sub_score_weight: int = 0
    final_score_weight: int = 1
    pred_ce_weight: int = 1
    pred_l1_weight: int = 0.1
    pred_area_weight: int = 2
    prev_weight: int = 0.1
    agent_class_weight: float = 1.0
    agent_box_weight: float = 0.1
    bev_semantic_weight: float = 1.0

    #others
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    lidar_architecture: str = "resnet34"

    # latent / world model
    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    # future latent prediction
    predict_future: bool = False
    future_frame_offsets: tuple = (1, 2)     # which future frames (t+k) to predict at train time
    cache_future_frame_offsets: tuple = ()   # superset stored in target cache; empty = no future in cache
    future_prediction_weight: float = 0.1    # loss weight in total loss
    future_prediction_type: str = "l1"      # "l1" | "mse" | "cosine"
    future_latent_space: str = "projected_map"  # "projected_map" (FL-1) | "pretrain_vit" (FL-2) | "lora_to_pretrain_vit" (FL-3)
    future_target_mode: str = "frozen_pretrain_vit"  # "frozen_pretrain_vit" | "ema_online_encoder"
    future_target_pair_mode: str = "repeat_future"  # "repeat_future" | "current_future"=[t,t+0.5]
    future_ema_decay: float = 0.99925        # V-JEPA 2.1-style EMA teacher momentum
    future_target_layernorm: bool = True    # LayerNorm on future target latent (FL-3)
    future_latent_bottleneck: int = 256      # conv predictor bottleneck when arch=conv
    future_predictor_arch: str = "conv"        # "conv" | "jepa_token"
    future_predictor_dim: int = 512            # jepa_token hidden dim
    future_predictor_depth: int = 4            # jepa_token transformer layers
    future_predictor_heads: int = 8            # jepa_token attention heads
    # GT-only latent supervision + scorer-only proposal rollout features.
    # Opt-in: legacy FL/SC-2 checkpoints and data flow remain unchanged.
    future_gt_trajectory_conditioned: bool = False
    future_rollout_layers: int = 2
    # === EMA-JQTF (isolated opt-in path) ===
    # Q3 jointly produces proposal^3 and proposal-conditioned future tokens.
    ema_jqtf_candidate_layers: int = 2
    ema_jqtf_full_decoder_layers: int = 2
    ema_jqtf_scorer_layers: int = 2
    ema_jqtf_external_adapter_layers: int = 1
    ema_jqtf_scorer_alpha_init: float = 0.0
    ema_jqtf_scorer_head_mode: str = "base_delta"  # "base_delta" | "direct_final" | "formula_progress"
    ema_jqtf_pdm_query_layers: int = 1
    ema_jqtf_formula_scorer_lr_scale: float = 3.0
    ema_jqtf_formula_future_lr_scale: float = 5.0
    ema_jqtf_formula_lr_milestone: int = 20
    ema_jqtf_formula_lr_gamma: float = 0.3
    # "detached": no score gradient upstream; "representation": connect
    # Q4/BEV/future but not trajectory coordinates; "full": connect all.
    ema_jqtf_scorer_gradient_mode: str = "detached"
    ema_jqtf_decode_selected_future: bool = True

    # future latent integration into planning/scoring
    future_integration: str = "none"           # "none" | "scorer_pool" | "ema_jqtf"
    run_future_at_inference: bool = False      # explicit future branch use during eval/navtest
    future_pool_fusion: str = "gate_add"       # "add" | "gate_add"
    future_scorer_detach: bool = False         # detach predicted future latent before Scorer fusion

    # inference-time counterfactual diagnostics for scorer_pool (eval only, no retrain)
    #   none        : use predicted future latent as-is (normal ON path)
    #   off         : drop future latent before Scorer (equivalent to no-future / gate off)
    #   shuffle     : permute predicted future latent across the batch (break scene correspondence)
    #   persistence : feed current-frame latent as the "future" (test whether prediction matters)
    future_diag_mode: str = "none"             # "none" | "off" | "shuffle" | "persistence"

    # === SC-2: trajectory-conditioned scorer (opt-in; default keeps legacy scorer) ===
    # 每条 proposal 编码轨迹几何 -> cross-attend 当前场景(+预测未来) token -> factorized PDM 分头。
    scorer_traj_conditioned: bool = False      # 打开 SC-2 分支
    scorer_xattn_layers: int = 2               # trajectory-token cross-attention 层数
    scorer_traj_residual: bool = True          # cross-attn 输出与 bev proposal_feature 残差相加
    scorer_use_future_scene: bool = True       # future_integration=scorer_pool 时把预测未来 token 加入 scene memory
    scorer_ranking_weight: float = 2.0         # pairwise margin ranking loss 权重（0 关闭）
    scorer_ranking_margin: float = 0.05        # margin ranking 边距
    scorer_pdm_formula_inference: bool = True  # 推理用 PDM 公式组合 subscore 选轨迹（否则用 final logit）
    # Dual-progress Formula mode predicts distances in normalized units, but
    # never clips them.  50 m is approximately the train-cache P99.
    scorer_progress_scale: float = 50.0
    # Used by the opt-in official_threshold Formula mode. Legacy symmetric
    # mode ignores it and remains numerically unchanged.
    scorer_progress_distance_threshold: float = 5.0
    scorer_formula_progress_mode: str = "symmetric"
    # Selection-only epsilon; the reported Formula score remains unchanged.
    scorer_formula_safe_progress_tie_break_weight: float = 0.0
    # Inference-only Formula candidate selection. ``current`` preserves the
    # trained baseline; the alternatives require no checkpoint changes.
    scorer_formula_selection_mode: str = "current"
    scorer_formula_safety_topk: int = 8
    # Formula-progress metric heads only. "smooth_l1" preserves V0 exactly;
    # "bce" applies BCEWithLogits and "focal" applies sigmoid Focal loss to
    # NOC/DAC/TTC/Comfort while leaving both progress regressions unchanged.
    scorer_formula_metric_loss_type: str = "smooth_l1"
    scorer_formula_focal_gamma: float = 2.0
    # Positive means safe for all four heads; alpha=0.25 therefore gives
    # failure targets a 3:1 class cost before difficulty modulation.
    scorer_formula_focal_alpha: float = 0.25
    # Relative sample weights for Formula BCE safety failures. Each weighted
    # head is normalized by its weight sum so the nominal head scale is stable.
    scorer_formula_noc_failure_weight: float = 1.0
    scorer_formula_noc_partial_weight: float = 1.0
    scorer_formula_ttc_failure_weight: float = 1.0
    # Generated-candidate Progress supervision. Defaults reproduce the legacy
    # Formula loss exactly.  Top-k is defined by the target EP score, not by
    # raw travelled metres or predicted safety.
    scorer_formula_raw_progress_loss_weight: float = 1.0
    scorer_formula_pdm_progress_loss_weight: float = 1.0
    scorer_progress_topk: int = 8
    scorer_progress_topk_weight: float = 1.0
    scorer_progress_ranking_weight: float = 0.0
    scorer_progress_ranking_margin_cap: float = 0.05
    scorer_progress_ranking_tie_epsilon: float = 1e-4
    # Optional local-epoch schedules. Empty tuples preserve static legacy
    # weights; after the final entry, the last value is held.
    scorer_loss_weight_schedule: Tuple[float, ...] = ()
    future_prediction_weight_schedule: Tuple[float, ...] = ()
    scorer_progress_ranking_in_bundle: bool = False
    scorer_ranking_in_bundle: bool = False
    scorer_safe_progress_tie_ranking_weight: float = 0.0
    scorer_safe_progress_tie_ranking_in_bundle: bool = False
    scorer_safe_progress_tie_safety_threshold: float = 0.95
    scorer_safe_progress_tie_ep_epsilon: float = 1e-4
    scorer_safe_progress_tie_raw_epsilon_m: float = 0.01
    # Training-only directional final-score correction. The inference output
    # remains one Direct-Final logit.
    scorer_asymmetric_safety_weight: float = 0.0
    scorer_false_safe_weight: float = 3.0
    scorer_false_unsafe_weight: float = 1.0
    scorer_asymmetric_safety_margin: float = 0.05
    scorer_safety_metric_threshold: float = 0.95
    scorer_safe_final_threshold: float = 0.9

    # === training-only hard-anchor pointwise scorer supervision ===
    # Default off: forward/inference and legacy losses remain numerically unchanged.
    scorer_anchor_auxiliary: bool = False
    anchor_trajectory_path: str = "./data/8192.npy"
    scorer_anchor_cache_root: str = ""          # empty -> NAVSIM_EXP_ROOT/Drive-JEPA-cache/anchors_scores_v1_full_fp16
    scorer_anchor_sampling: str = "hard_balanced"  # "hard_balanced" | "balanced" | "formula_counterfactual"
    scorer_anchor_num_local_hard: int = 64
    scorer_anchor_num_gt_local_hard: int = 0
    scorer_anchor_num_balanced: int = 64
    scorer_anchor_nearest_per_proposal: int = 64
    scorer_anchor_nearest_per_gt: int = 64
    scorer_anchor_min_score_gap: float = 0.2
    scorer_anchor_yaw_weight: float = 0.5
    scorer_anchor_loss_weight: float = 0.5
    scorer_anchor_subscore_weight: float = 1.0
    scorer_anchor_final_weight: float = 1.0
    scorer_anchor_seed: int = 20260715
    # Formula-only scorer augmentation. The 128 cached trajectories are
    # point-supervised; no ranking loss is implied by this switch.
    scorer_formula_counterfactual: bool = False
    scorer_formula_cf_num_ttc_only: int = 24
    scorer_formula_cf_num_joint_unsafe: int = 8
    scorer_formula_cf_num_safe_rescue: int = 16
    scorer_formula_cf_num_safe_progress: int = 16
    scorer_formula_cf_num_balanced: int = 64
    scorer_formula_cf_safe_metric_min: float = 0.95
    scorer_formula_cf_failure_metric_max: float = 0.05
    scorer_formula_cf_safe_final_min: float = 0.90
    scorer_formula_cf_progress_high_min: float = 0.80
    scorer_formula_cf_progress_low_max: float = 0.60
    scorer_only_training: bool = False
    # Load model weights from checkpoint_path, but create a fresh optimizer and
    # train the complete joint recipe. This is intentionally distinct from
    # Lightning resume, which would also restore old optimizer/trainer state.
    joint_continuation_training: bool = False

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # detection
    num_bounding_boxes: int = 30

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
