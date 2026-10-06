from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class DriveJEPAConfig:
    # V-JEPA2 lives outside the NAVSIM repository in the original workspace.
    # These may be overridden by Hydra; empty values fall back to VJEPA2_ROOT
    # and NAVSIM_EXP_ROOT in ``ImgEncoder``.
    vjepa2_root: str = ""
    vjepa_checkpoint_path: str = ""
    vjepa_config_path: str = ""

    b2d: bool = False

    ref_num: int=4

    traj_bev: bool=True
    traj_proposal_query: bool=True

    double_score: bool=False

    # Legacy auxiliary labels depended on private scorer internals.  Keep them
    # disabled until equivalent official-current target builders are provided.
    agent_pred: bool=False
    area_pred: bool=False

    bev_map: bool=False
    bev_agent: bool=False

    proposal_num: int = 32

    # NAVSIM v2 scorer / migration controls.  The baseline deliberately starts
    # with anchors and ranking disabled; enable them as single-variable phases.
    direct_epdms_head: bool = False
    # V2-native SC-3 migration: a proposal-conditioned rollout transformer is
    # trained with the scorer, while the V2 generator stays frozen.
    sc3_rollout_enabled: bool = False
    sc3_rollout_layers: int = 2
    sc3_rollout_heads: int = 8
    sc3_rollout_dropout: float = 0.0
    scorer_only: bool = False
    use_external_anchors: bool = False
    external_anchor_weight: float = 0.5
    external_anchor_path: str = ""
    external_anchor_cache_root: str = ""
    external_anchor_num_local_hard: int = 64
    external_anchor_num_ttc_no_collision: int = 0
    external_anchor_num_balanced: int = 64
    external_anchor_nearest_per_proposal: int = 64
    external_anchor_min_score_gap: float = 0.20
    external_anchor_metric_failure_threshold: float = 0.90
    external_anchor_high_safe_threshold: float = 0.95
    external_anchor_ttc_failure_max: float = 0.05
    external_anchor_yaw_weight: float = 0.5
    external_anchor_seed: int = 20260723
    ranking_weight: float = 0.0
    ranking_margin: float = 0.02
    ranking_min_return_gap: float = 0.02
    ranking_quantile_edges: Tuple[float, ...] = ()
    ranking_category_weights: Tuple[float, ...] = (0.45, 0.20, 0.25, 0.10)

    # Only train splits may provide scorer labels.  Test/warmup/private labels
    # are rejected before a training worker is created.
    training_split_name: str = "navtrain"
    metric_cache_path: str = ""
    score_workers: int = 16

    # Every current tensor is still required. Only the two explicitly named
    # retired auxiliary heads may be ignored from the completed v2 checkpoint.
    # A v1 warm-start must explicitly name compatible modules.
    checkpoint_load_policy: str = "strict_ignore"
    checkpoint_report_path: str = ""
    checkpoint_ignored_prefixes: Tuple[str, ...] = (
        "_pad_model.scorer.pred_col_agent",
        "_pad_model.scorer.pred_area",
    )
    checkpoint_allowed_missing_prefixes: Tuple[str, ...] = (
        "_pad_model.scorer.proposal_rollout",
        "_pad_model.scorer.proposal_rollout_norm",
        "_pad_model.scorer.proposal_rollout_gate",
    )
    warm_start_prefixes: Tuple[str, ...] = (
        "_pad_model._backbone",
        "_pad_model._trajectory_head",
    )
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
    image_architecture: str = "resnet34"

    # loss weights
    trajectory_weight: float = 1
    inter_weight: float =  0
    sub_score_weight: int = 1
    final_score_weight: int = 1
    direct_score_weight: float = 0.0
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

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

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
