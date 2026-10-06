"""Dependency-light command mapping and trajectory controller helpers."""

import math

import numpy as np


_NAVSIM_COMMAND_TO_INDEX = {
    "LEFT": 0,
    "CHANGELANELEFT": 0,
    "STRAIGHT": 1,
    "LANEFOLLOW": 1,
    "RIGHT": 2,
    "CHANGELANERIGHT": 2,
}


_BENCH2DRIVE_COMMAND_TO_INDEX = {
    "LEFT": 0,
    "RIGHT": 1,
    "STRAIGHT": 2,
    "LANEFOLLOW": 3,
    "CHANGELANELEFT": 4,
    "CHANGELANERIGHT": 5,
}


def road_option_to_one_hot(option, command_num=4):
    """Map CARLA RoadOption to the checkpoint's command vocabulary."""
    name = getattr(option, "name", str(option)).upper().split(".")[-1]
    if int(command_num) == 4:
        output = [0.0] * 4
        output[_NAVSIM_COMMAND_TO_INDEX.get(name, 3)] = 1.0
        return output
    if int(command_num) != 6:
        raise ValueError("command_num must be 4 or 6")
    output = [0.0] * 6
    output[_BENCH2DRIVE_COMMAND_TO_INDEX.get(name, 3)] = 1.0
    return output


def validate_trajectory(trajectory):
    array = np.asarray(trajectory, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("trajectory must have shape [N>=2, 2 or 3]")
    if not np.isfinite(array).all():
        raise ValueError("trajectory contains non-finite values")
    return array[:, :3] if array.shape[1] >= 3 else array[:, :2]


def trajectory_to_control(trajectory, current_speed, state, config):
    """Convert an ego-local NAVSIM trajectory (x forward, y left) to controls."""
    points = validate_trajectory(trajectory)
    xy = points[:, :2]

    lookahead_mode = str(config.get("lookahead_mode", "fixed_index"))
    if lookahead_mode == "speed_distance":
        lookahead_distance = float(
            np.clip(
                float(config.get("lookahead_speed_scale", 0.5))
                * float(current_speed)
                + float(config.get("lookahead_distance_offset", 2.5)),
                float(config.get("lookahead_distance_min", 4.0)),
                float(config.get("lookahead_distance_max", 8.0)),
            )
        )
        # SparseDrive V2 chooses the path point whose ego distance is closest
        # to a speed-dependent preview distance.  Our path is temporal rather
        # than spatial, but the same selection removes the fixed-time change
        # in physical preview distance.
        lookahead_index = int(
            np.argmin(np.abs(np.linalg.norm(xy, axis=1) - lookahead_distance))
        )
    elif lookahead_mode == "fixed_index":
        lookahead_index = min(int(config.get("lookahead_index", 2)), len(xy) - 1)
        lookahead_distance = float(np.linalg.norm(xy[lookahead_index]))
    else:
        raise ValueError(f"unsupported lookahead_mode: {lookahead_mode}")
    target = xy[lookahead_index]
    heading = math.atan2(float(target[1]), max(float(target[0]), 0.1))
    max_heading = max(float(config.get("max_heading_radians", 0.7)), 1e-3)
    desired_steer = float(config.get("steer_sign", -1.0)) * heading / max_heading
    desired_steer *= float(config.get("steer_gain", 1.0))
    desired_steer = float(np.clip(desired_steer, -1.0, 1.0))

    old_steer = float(state.get("steer", 0.0))
    steer_rate = float(config.get("steer_rate_limit", 0.18))
    steer = float(np.clip(desired_steer, old_steer - steer_rate, old_steer + steer_rate))

    origin = np.zeros((1, 2), dtype=np.float32)
    segments = np.linalg.norm(
        np.diff(np.concatenate([origin, xy], axis=0), axis=0), axis=1
    )
    speed_points = max(1, min(int(config.get("speed_points", 4)), len(segments)))
    interval = max(float(config.get("trajectory_interval", 0.5)), 1e-3)
    speed_estimator = str(config.get("speed_estimator", "median_from_origin"))
    if speed_estimator == "median_from_origin":
        speed_distance = float(np.median(segments[:speed_points]))
    elif speed_estimator == "mean_interpoint":
        # iPad estimates speed from the mean displacement between predicted
        # waypoints.  It deliberately excludes the origin-to-first-point leg.
        interpoint = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        count = max(1, min(speed_points, len(interpoint)))
        speed_distance = float(np.mean(interpoint[:count]))
    else:
        raise ValueError(f"unsupported speed_estimator: {speed_estimator}")
    target_speed = float(speed_distance / interval)
    target_speed = float(
        np.clip(target_speed, 0.0, float(config.get("max_target_speed", 12.0)))
    )

    speed_error = target_speed - float(current_speed)
    if speed_error <= 0.0:
        # Do not retain positive throttle wind-up while asking the car to slow.
        integral = 0.0
    else:
        integral = float(state.get("speed_integral", 0.0)) + speed_error * float(
            config.get("control_interval", 0.05)
        )
    integral = float(np.clip(integral, -10.0, 10.0))
    throttle = float(
        np.clip(
            float(config.get("speed_kp", 0.35)) * speed_error
            + float(config.get("speed_ki", 0.03)) * integral,
            0.0,
            float(config.get("max_throttle", 0.75)),
        )
    )
    brake = float(
        np.clip(-float(config.get("brake_kp", 0.45)) * speed_error, 0.0, 1.0)
    )
    if target_speed < float(config.get("stop_target_speed", 0.05)):
        throttle = 0.0
        brake = max(brake, float(config.get("stop_brake", 0.7)))
    elif float(current_speed) < float(config.get("creep_speed_threshold", 0.3)):
        # Parking-exit predictions often request a small positive initial speed.
        # Ensure that intent can overcome CARLA vehicle static friction.
        throttle = max(throttle, float(config.get("min_start_throttle", 0.18)))
        brake = 0.0

    # CARLA accepts both channels, but iPad and SparseDrive V2 make them
    # mutually exclusive.  This also prevents stale PI state from fighting a
    # new braking request.
    if brake > 0.0:
        throttle = 0.0

    return {
        "steer": steer,
        "throttle": throttle,
        "brake": brake,
        "target_speed": target_speed,
        "heading": heading,
        "lookahead_index": lookahead_index,
        "lookahead_distance": lookahead_distance,
        "state": {"steer": steer, "speed_integral": integral},
    }


def trajectory_to_world(trajectory, location, forward, right):
    """Freeze a NAVSIM-local trajectory into CARLA world xy coordinates."""
    points = validate_trajectory(trajectory)
    world = []
    for point in points:
        world.append(
            [
                float(location.x + point[0] * forward.x - point[1] * right.x),
                float(location.y + point[0] * forward.y - point[1] * right.y),
            ]
        )
    return np.asarray(world, dtype=np.float32)


def world_to_local_trajectory(world_points, location, forward, right):
    """Re-express cached world points in the current NAVSIM ego frame."""
    world = np.asarray(world_points, dtype=np.float32)
    dx = world[:, 0] - float(location.x)
    dy = world[:, 1] - float(location.y)
    x_forward = dx * float(forward.x) + dy * float(forward.y)
    y_left = -(dx * float(right.x) + dy * float(right.y))
    return np.stack([x_forward, y_left], axis=-1)
