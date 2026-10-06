import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.creation import linestrings
from shapely import Point, creation
from shapely.strtree import STRtree
import shapely
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.observation.idm.utils import is_agent_behind, is_track_stopped
from shapely import LineString, Polygon
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    EgoAreaIndex,
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)
from ..ema_jqtf.formula_selection import formula_progress_ratio
from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _candidate_comfort(trajectories, interval_s=0.5):
    """Approximate NAVSIM comfort bounds for sampled [x, y, yaw] poses."""
    trajectories = np.asarray(trajectories, dtype=np.float64)
    origin = np.zeros((*trajectories.shape[:-2], 1, 3), dtype=trajectories.dtype)
    poses = np.concatenate([origin, trajectories], axis=-2)
    velocity = np.diff(poses[..., :2], axis=-2) / interval_s
    acceleration = np.diff(velocity, axis=-2) / interval_s
    jerk = np.diff(acceleration, axis=-2) / interval_s
    yaw = np.unwrap(poses[..., 2], axis=-1)
    yaw_rate = np.diff(yaw, axis=-1) / interval_s
    yaw_accel = np.diff(yaw_rate, axis=-1) / interval_s

    accel_ok = (np.linalg.norm(acceleration, axis=-1) < 4.89).all(axis=-1)
    jerk_ok = (np.linalg.norm(jerk, axis=-1) < 8.37).all(axis=-1)
    yaw_rate_ok = (np.abs(yaw_rate) < 0.95).all(axis=-1)
    yaw_accel_ok = (np.abs(yaw_accel) < 1.93).all(axis=-1)
    return accel_ok & jerk_ok & yaw_rate_ok & yaw_accel_ok


def _normalize_scorer_cache(cache):
    if isinstance(cache, dict) and "samples" in cache:
        return cache["samples"]
    return cache


def b2d_before_score(map_infos, proposals, targets, scorer_cache, config):
    """Prepare per-scene candidate geometry for the B2D scorer.

    ``scorer_cache`` is produced by ``bench2drive.training.scorer_cache``.
    Candidate-dependent DAC/route masks are intentionally calculated here,
    after the model has generated its trajectories.
    """
    samples = _normalize_scorer_cache(scorer_cache)
    target_trajectory = targets["trajectory"]
    proposals_np = _as_numpy(proposals)
    target_np = _as_numpy(target_trajectory)
    tokens = list(targets["token"])
    interval_s = float(config.trajectory_sampling.interval_length)
    data_points = []

    for batch_index, token_value in enumerate(tokens):
        token = str(token_value)
        if token not in samples:
            raise KeyError(
                f"Bench2Drive scorer cache has no token {token!r}; "
                "regenerate it with the same dataset token policy"
            )
        sample = samples[token]
        proposal = proposals_np[batch_index]
        expert = target_np[batch_index]
        if proposal.shape[-2:] != expert.shape[-2:]:
            raise ValueError(
                f"proposal/target trajectory mismatch for {token}: "
                f"{proposal.shape} vs {expert.shape}"
            )

        # Append the expert only as a collision reference. get_sub_score drops
        # this last trajectory from the returned candidate labels.
        all_trajectories = np.concatenate([proposal, expert[None]], axis=0)
        all_ttc = []
        xy = all_trajectories[..., :2]
        velocity = np.concatenate([xy[..., :1, :], np.diff(xy, axis=-2)], axis=-2) / interval_s
        for offset_s in (0.0, 0.5, 1.0):
            extrapolated = all_trajectories.copy()
            extrapolated[..., :2] += velocity * offset_s
            all_ttc.append(extrapolated)
        trajectories_ttc = np.stack(all_ttc, axis=-2)
        ego_corners = compute_corners_torch(
            torch.from_numpy(trajectories_ttc).float(),
            half_length=float(config.half_length),
            half_width=float(config.half_width),
            rear_axle_to_center=float(config.rear_axle_to_center),
        ).numpy()

        world2ego = np.asarray(sample["world2ego"], dtype=np.float64)
        ego2world = np.linalg.inv(world2ego)
        center = np.concatenate(
            [xy, np.zeros((*xy.shape[:-1], 1)), np.ones((*xy.shape[:-1], 1))],
            axis=-1,
        )
        corner_xyz1 = np.concatenate(
            [
                ego_corners[..., 0, :, :],
                np.zeros((*ego_corners[..., 0, :, :].shape[:-1], 1)),
                np.ones((*ego_corners[..., 0, :, :].shape[:-1], 1)),
            ],
            axis=-1,
        )
        world_centers = np.einsum("ij,ptj->pti", ego2world, center)[..., :2]
        world_corners = np.einsum("ij,ptkj->ptki", ego2world, corner_xyz1)[..., :2]

        town_name = str(sample["town_name"])
        if town_name not in map_infos:
            raise KeyError(f"HD-map cache has no entry for {town_name!r}")
        lanes = np.asarray(map_infos[town_name], dtype=np.float64)
        if lanes.ndim != 2 or lanes.shape[1] < 4:
            raise ValueError(
                f"HD-map entry {town_name} must be [N,>=4] "
                "with x,y,half_width,road_id"
            )
        lane_xy = lanes[:, :2]
        lane_half_width = lanes[:, 2]
        lane_road_id = lanes[:, 3]

        flat_centers = world_centers.reshape(-1, 2)
        midpoint = (flat_centers.max(axis=0) + flat_centers.min(axis=0)) / 2.0
        radius = np.linalg.norm(flat_centers - midpoint, axis=-1).max() + 8.0
        nearby = np.linalg.norm(lane_xy - midpoint, axis=-1) <= radius
        if not nearby.any():
            on_road = np.zeros(world_centers.shape[:-1], dtype=np.bool_)
            on_route = np.zeros_like(on_road)
            multiple_lanes = np.zeros_like(on_road)
        else:
            lane_xy = lane_xy[nearby]
            lane_half_width = lane_half_width[nearby]
            lane_road_id = lane_road_id[nearby]
            corner_distance = np.linalg.norm(
                world_corners[None] - lane_xy[:, None, None, None], axis=-1
            )
            within_lane = corner_distance <= lane_half_width[:, None, None, None]
            on_road = within_lane.any(axis=0).all(axis=-1)

            center_distance = np.linalg.norm(
                world_centers[None] - lane_xy[:, None, None], axis=-1
            ) - lane_half_width[:, None, None]
            nearest = center_distance.argmin(axis=0)
            nearest_road = lane_road_id[nearest]
            expert_road_ids = np.unique(nearest_road[-1])
            on_route = np.isin(nearest_road, expert_road_ids)

            corner_nearest = corner_distance.argmin(axis=0)
            corner_roads = lane_road_id[corner_nearest]
            multiple_lanes = (corner_roads != corner_roads[..., :1]).any(axis=-1)

        ego_areas = np.stack([multiple_lanes, on_road, on_route], axis=-1)
        future_boxes = np.asarray(sample["future_box_corners"], dtype=np.float32)
        future_valid = np.asarray(sample["future_box_valid"], dtype=np.bool_)
        if future_boxes.shape[1] != proposal.shape[1]:
            raise ValueError(
                f"future-box horizon mismatch for {token}: "
                f"{future_boxes.shape[1]} vs {proposal.shape[1]}"
            )
        future_boxes = future_boxes.copy()
        future_boxes[~future_valid] = 0.0

        data_points.append(
            {
                "fut_box_corners": future_boxes,
                "_ego_coords": ego_corners,
                "proposal": proposal,
                "target_traj": expert,
                "comfort": _candidate_comfort(proposal, interval_s=interval_s),
                "ego_areas": ego_areas,
                "pdm_progress": float(sample["expert_progress"]),
            }
        )
    return data_points

def compute_corners_torch(
        proposals,
        half_length=2.042,
        half_width=0.925,
        rear_axle_to_center=0.39,
):

    headings= proposals[...,2]
    cos_yaw = torch.cos(headings)
    sin_yaw = torch.sin(headings)

    x = proposals[...,0]+rear_axle_to_center*cos_yaw
    y = proposals[...,1]+rear_axle_to_center*sin_yaw
    half_length=half_length+torch.zeros_like(headings)
    half_width=half_width+torch.zeros_like(headings)

    cos_yaw=cos_yaw[...,None]
    sin_yaw=sin_yaw[...,None]

    # Compute the four corners
    corners_x = torch.stack([half_length, -half_length, -half_length, half_length],dim=-1)
    corners_y = torch.stack([half_width, half_width, -half_width, -half_width],dim=-1)

    # Rotate corners by yaw
    rot_corners_x = cos_yaw * corners_x + (-sin_yaw) * corners_y
    rot_corners_y = sin_yaw * corners_x + cos_yaw * corners_y

    # Translate corners to the center of the bounding box
    corners = torch.stack((rot_corners_x + x[...,None], rot_corners_y + y[...,None]), dim=-1)

    return corners
    # FRONT_LEFT = 0
    # REAR_LEFT = 1
    # REAR_RIGHT = 2
    # FRONT_RIGHT = 3


def get_collision_type(
        state,
        ego_polygon: Polygon,
        tracked_object_polygon: Polygon,
        track_speed,
        track_heading,
        stopped_speed_threshold: float = 5e-02,
):
    ego_speed = state[-1]

    is_ego_stopped = float(ego_speed) <= stopped_speed_threshold

    center_point = tracked_object_polygon.centroid

    tracked_object_center = StateSE2(center_point.x, center_point.y, track_heading)

    x=state[0]
    y=state[1]
    ego_heading=state[2]

    ego_rear_axle_pose: StateSE2 = StateSE2(x,y,ego_heading)

    # Collisions at (close-to) zero ego speed
    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION

    # Collisions at (close-to) zero track speed
    elif track_speed <= stopped_speed_threshold:
        collision_type = CollisionType.STOPPED_TRACK_COLLISION

    # Rear collision when both ego and track are not stopped
    elif is_agent_behind(ego_rear_axle_pose, tracked_object_center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION

    # Front bumper collision when both ego and track are not stopped
    elif LineString(
            [
                ego_polygon.exterior.coords[0],
                ego_polygon.exterior.coords[3],
            ]
    ).intersects(tracked_object_polygon):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION

    # Lateral collision when both ego and track are not stopped
    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION

    return collision_type


def evaluate_coll( fut_box_corners,_ego_coords,_ego_areas):
    n_future = _ego_coords.shape[1]
    _num_proposals=_ego_coords.shape[0]
    fut_mask=fut_box_corners.any(-1).any(-1)

    node_capacity=10

    _ego_polygons = shapely.creation.polygons(_ego_coords)

    proposal_fault_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }
    proposal_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }
    temp_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }
    ttc_collided_track_ids = {
        proposal_idx:[]
        for proposal_idx in range(_num_proposals)
    }

    key_agent_corners = np.zeros([_num_proposals, n_future, 2, 4, 2])
    key_agent_labels = np.zeros([_num_proposals, n_future, 2],dtype=bool)
    collision_all = np.zeros([_num_proposals,n_future],dtype=bool)
    ttc_collision_all = np.zeros([_num_proposals,n_future],dtype=bool)

    # ego_pos=_ego_coords[:,:,0].mean(-2)
    # ego_vel=(_ego_coords[:,:,1,0]-_ego_coords[:,:,0,0])*2
    #
    # speeds=np.linalg.norm(ego_vel,axis=-1)
    #
    # # Compute the vector along the front edge
    # front_edge = _ego_coords[:,:, 0,0, :] - _ego_coords[:,:, 0, 1, :]
    #
    # # Compute heading angle relative to x-axis
    # ego_headings = np.arctan2(front_edge[..., 1], front_edge[..., 0])
    #
    # ego_state=np.concatenate([ego_pos,ego_headings[...,None],speeds[...,None]],axis=-1)

    # fut_box_center=fut_box_corners.mean(-2)

    # track_object_vel = (fut_box_center[:,1:]-fut_box_center[:,:-1])*2
    #
    # track_object_vel=np.concatenate([track_object_vel[:,:1],track_object_vel],axis=1)
    #
    # track_object_speeds=np.linalg.norm(track_object_vel,axis=-1)

    # Compute the vector along the front edge
    # object_front_edge = fut_box_corners[..., 0, :] - fut_box_corners[..., 1, :]
    #
    # # Compute heading angle relative to x-axis
    # track_object_headings = np.arctan2(object_front_edge[..., 1], object_front_edge[..., 0])

    for time_idx in range(n_future):
        geometries=fut_box_corners[:,time_idx][fut_mask[:,time_idx]]
        _geometries = [Polygon(geometry) for geometry in geometries]
        _str_tree = STRtree(_geometries, node_capacity)
        ego_polygons = _ego_polygons[:, time_idx,0]

        intersecting = _str_tree.query(ego_polygons, predicate="intersects")

        token_list=np.arange(len(fut_box_corners))[fut_mask[:,time_idx]]

        for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
            token=token_list[geometry_idx]
            if token in proposal_collided_track_ids[proposal_idx] or len(proposal_fault_collided_track_ids[proposal_idx]):
                continue

            # ego_in_multiple_lanes_or_nondrivable_area = (
            #         _ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
            #         or _ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
            # )
            #
            # tracked_object_polygon = _geometries[geometry_idx]
            #
            # # classify collision
            # collision_type: CollisionType = get_collision_type(
            #     ego_state[proposal_idx][time_idx],
            #     ego_polygons[proposal_idx],
            #     tracked_object_polygon,
            #     track_object_speeds[token][time_idx],
            #     track_object_headings[token][time_idx],
            #
            # )
            # collisions_at_stopped_track_or_active_front: bool = collision_type in [
            #     CollisionType.ACTIVE_FRONT_COLLISION,
            #     CollisionType.STOPPED_TRACK_COLLISION,
            # ]
            # collision_at_lateral: bool = collision_type == CollisionType.ACTIVE_LATERAL_COLLISION

            # 1. at fault collision
            # if collisions_at_stopped_track_or_active_front or (
            #         ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
            # ):
            proposal_fault_collided_track_ids[proposal_idx].append(token)
            collision_all[proposal_idx][time_idx]=True
            key_agent_labels[proposal_idx][:time_idx+1,0]=fut_mask[token][:time_idx+1]
            key_agent_corners[proposal_idx][:time_idx+1,0]=fut_box_corners[token][:time_idx+1]
            # else:  # 2. no at fault collision
            #     proposal_collided_track_ids[proposal_idx].append(token)

        for ttc in [1,2]:
            ego_polygons = _ego_polygons[:, time_idx,ttc]

            intersecting = _str_tree.query(ego_polygons, predicate="intersects")

            for proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                token = token_list[geometry_idx]
                if (token in temp_collided_track_ids[proposal_idx] or len(ttc_collided_track_ids[proposal_idx]) #or (speeds[proposal_idx, time_idx] < 5e-02)
                ):
                    continue

                # ego_in_multiple_lanes_or_nondrivable_area = (
                #         _ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                #         or _ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                # )
                #
                # state=ego_state[proposal_idx][time_idx]
                # ego_heading = np.arctan2(state[-1], state[-2])
                #
                # ego_rear_axle: StateSE2 = StateSE2(state[0], state[1], ego_heading)
                # tracked_object_polygon = _geometries[geometry_idx]
                #
                # centroid = tracked_object_polygon.centroid
                # track_heading = track_object_headings[token][time_idx]
                # track_state = StateSE2(centroid.x, centroid.y, track_heading)

                # TODO: fix ego_area for intersection
                # if is_agent_ahead(ego_rear_axle, track_state) or (
                #         ego_in_multiple_lanes_or_nondrivable_area
                #         and not is_agent_behind(ego_rear_axle, track_state)
                # ):
                ttc_collided_track_ids[proposal_idx].append(token)
                ttc_collision_all[proposal_idx][time_idx]=True
                key_agent_labels[proposal_idx][:time_idx+1,1]=fut_mask[token][:time_idx+1]
                key_agent_corners[proposal_idx][:time_idx+1,1]=fut_box_corners[token][:time_idx+1]
                # else:
                #     temp_collided_track_ids[proposal_idx].append(token)

    return collision_all,ttc_collision_all,key_agent_corners[:-1],key_agent_labels[:-1]

def get_scores(args):

    return [
        get_sub_score(
            a["fut_box_corners"],
            a["_ego_coords"],
            a["proposal"],
            a["target_traj"],
            a["comfort"],
            a["ego_areas"],
            a["pdm_progress"],
        )
        for a in args
    ]

def get_sub_score(
        fut_box_corners,
        _ego_coords,
        proposals,
        target_traj,
        comfort,
        ego_areas,
        pdm_progress=None,
):

    collsions,ttc_collision,key_agent_corners,key_agent_labels=evaluate_coll(fut_box_corners,_ego_coords,ego_areas)

    collsions=collsions[:-1] & (~collsions[-1:])  #collsion=True and gt_collision =False

    ttc_collision=ttc_collision[:-1] & (~ttc_collision[-1:])  #collsion=True and gt_collision =False

    collision=1-collsions.any(-1)

    ttc=1-ttc_collision.any(-1)

    on_road_all=ego_areas[:-1,:,1]
    on_route_all=ego_areas[:-1,:,2]

    drivable_area_compliance=on_road_all.all(-1) & on_route_all.any(-1)

    ego_areas=np.stack([on_road_all,on_route_all],axis=-1)

    #l2 = np.linalg.norm(proposals[:,:, :2] - target_traj[None, : ,:2], axis=-1).mean(-1)

    # l2_score=l2-100*multiplicate_metric_scores
    #
    # sort_score=np.sort(l2_score)
    #
    # progress =np.zeros([len(proposals)])
    #
    # progress[sort_score[0]==l2_score]=1
    # progress[sort_score[1]==l2_score]=1

    target_line=np.concatenate([np.zeros([1,2]),target_traj[...,:2]])

    centerline=linestrings(target_line)

    target_progress = centerline.project(Point(target_line[-1]))

    raw_progress = np.ones([len(proposals)])

    for proposal_idx,proposal in enumerate(proposals[...,:2]):
        end_point = Point(proposal[-1])
        proj_progress=centerline.project(end_point)
        if proj_progress==target_progress:
            proj_progress=proj_progress+np.linalg.norm(proposal[-1]-target_traj[-1][:2])

        raw_progress[proposal_idx] = proj_progress #clip by max target progress

    raw_progress = np.clip(raw_progress, a_min=0, a_max=None)

    multiplicate_metric_scores=collision*drivable_area_compliance

    if pdm_progress is None:
        pdm_progress = target_progress
    progress_ratio = formula_progress_ratio(raw_progress, pdm_progress)
    progress=multiplicate_metric_scores*progress_ratio
    #raw_progress=multiplicate_metric_scores*raw_progress

    # proposals_xy=np.concatenate([np.zeros_like(proposals[:,:1,:2]),proposals[:,:,:2]],axis=1)
    #
    # vel=(proposals_xy[:,1:,:2]-proposals_xy[:,:-1,:2])/0.5
    #
    # acc=np.linalg.norm(vel[:,1:]-vel[:,:-1],axis=-1)/0.5

    # angle=np.arctan2(vel[:,:,1], vel[:,:,0])

    # heading=np.concatenate([np.zeros_like(angle[:,:1])+np.pi/2,angle],axis=1)

    # yaw_rate=(heading[:,1:]-heading[:,:-1])/0.5
    
    # yaw_accel=(yaw_rate[:,1:]-yaw_rate[:,:-1])/0.5

    # desired_speed=np.linalg.norm(vel,axis=-1).mean(-1)
    
    # comfort=(acc<10).all(-1) #& (desired_speed<15) & (np.abs(yaw_rate)<2).all(-1) & (np.abs(yaw_accel)<4).all(-1)


    final_scores=collision*drivable_area_compliance*(ttc*5/12+progress*5/12+comfort*2/12)

    target_scores=np.stack([collision,drivable_area_compliance,progress,ttc,comfort,final_scores],axis=-1)

    #print(target_scores[0])
    # if target_scores.mean()!=1:
    #     print(1)

    scores_index = np.empty((0,), dtype=np.int64)
    return (
        target_scores,
        key_agent_corners,
        key_agent_labels,
        ego_areas,
        raw_progress.astype(np.float32),
        np.float32(pdm_progress),
        scores_index,
    )
