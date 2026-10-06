"""Bench2Drive sensor agent backed by the Drive-JEPA model service."""

import collections
from camera_contract import CAMERA_ORDER, camera_sensors, validate_camera_contract
import json
import os
import socket
import time

import carla
import cv2
import numpy as np

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from bench2drive.bridge_protocol import PROTOCOL_VERSION, receive_message, send_message
from bench2drive.controller import (
    road_option_to_one_hot,
    trajectory_to_control,
    trajectory_to_world,
    world_to_local_trajectory,
)


def get_entry_point():
    return "MultiViewBench2DriveAgent"


class MultiViewBench2DriveAgent(AutonomousAgent):
    """Keep the CARLA Python 3.8 process separate from Drive-JEPA Python 3.9."""

    def setup(self, path_to_conf_file):
        self._socket = None
        # Bench2Drive appends route metadata to TEAM_CONFIG with a plus sign.
        config_path = path_to_conf_file.partition("+")[0]
        with open(config_path, "r", encoding="utf-8") as stream:
            self.config = json.load(stream)

        validate_camera_contract(self.config.get("features", {}))
        self.track = Track.SENSORS
        bridge = self.config["bridge"]
        self._host = bridge.get("host", "127.0.0.1")
        self._port = int(
            os.environ.get("DRIVE_JEPA_BRIDGE_PORT", bridge.get("port", 50123))
        )
        self._timeout = float(bridge.get("timeout_seconds", 60.0))
        self._connect_timeout = float(bridge.get("connect_timeout_seconds", 30.0))
        self._inference_interval = float(
            bridge.get("inference_interval_seconds", 0.5)
        )
        self._history_seconds = float(bridge.get("history_seconds", 0.5))
        self._jpeg_quality = int(bridge.get("jpeg_quality", 90))
        self._diagnostic_interval_steps = int(
            bridge.get("diagnostic_interval_steps", 40)
        )
        self._command_num = int(
            self.config.get("ego_status", {}).get("command_num", 4)
        )
        self._velocity_mode = str(
            self.config.get("ego_status", {}).get(
                "velocity_mode", "longitudinal_lateral"
            )
        )
        if self._velocity_mode not in ("scalar_speed", "longitudinal_lateral"):
            raise ValueError("unsupported ego_status.velocity_mode")
        if self._command_num not in (4, 6):
            raise ValueError("ego_status.command_num must be 4 or 6")

        self._frames = collections.deque(maxlen=40)
        self._last_inference_time = None
        self._world_trajectory = None
        self._controller_state = {}
        self._last_diagnostics = {}
        self._step = 0
        self._route_cursor = 0

    def sensors(self):
        return camera_sensors() + [dict(type='sensor.speedometer', reading_frequency=20, id='speed')]

    def _connect(self):
        if self._socket is not None:
            return
        deadline = time.time() + self._connect_timeout
        last_error = None
        while time.time() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self._timeout)
            try:
                sock.connect((self._host, self._port))
                send_message(
                    sock,
                    {"version": PROTOCOL_VERSION, "op": "ping", "client": "bench2drive"},
                )
                response = receive_message(sock)
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "model server ping failed"))
                self._socket = sock
                print(
                    "Drive-JEPA bridge connected to {}:{} ({})".format(
                        self._host, self._port, response.get("model", "unknown model")
                    ),
                    flush=True,
                )
                return
            except Exception as error:
                last_error = error
                sock.close()
                time.sleep(0.5)
        raise RuntimeError(
            "could not connect to Drive-JEPA model server: {}".format(last_error)
        )

    @staticmethod
    def _camera_jpeg(sensor_array, quality):
        array = np.asarray(sensor_array)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("front camera did not return a BGRA image")
        bgr = np.ascontiguousarray(array[:, :, :3])
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("failed to encode front camera frame")
        return encoded.tobytes()

    def _nearest_command(self):
        plan = self._global_plan_world_coord or []
        if not plan or self.hero_actor is None:
            return road_option_to_one_hot("LANEFOLLOW", self._command_num)
        location = self.hero_actor.get_location()
        nearest = min(
            range(len(plan)),
            key=lambda index: plan[index][0].location.distance(location),
        )
        option = plan[min(nearest + 1, len(plan) - 1)][1]
        return road_option_to_one_hot(option, self._command_num)

    def _navigation_target(self):
        """Mirror Bench2Drive RoutePlanner(4 m, 50 m) in world coordinates."""
        plan = self._global_plan_world_coord or []
        if not plan or self.hero_actor is None:
            return [0.0, 0.0], False, None
        last = len(plan)-1
        self._route_cursor = min(self._route_cursor, last)
        location = self.hero_actor.get_location()
        to_pop = 0
        farthest = float('-inf')
        cumulative = 0.0
        for offset in range(1, len(plan)-self._route_cursor):
            if cumulative > 50.0:
                break
            previous = plan[self._route_cursor+offset-1][0].location
            point = plan[self._route_cursor+offset][0].location
            cumulative += previous.distance(point)
            distance = location.distance(point)
            if distance <= 4.0 and distance > farthest:
                farthest = distance
                to_pop = offset
        self._route_cursor = min(self._route_cursor+to_pop, max(last-1, 0))
        target_index = min(self._route_cursor+1, last)
        target, option = plan[target_index]
        delta_x = float(target.location.x-location.x)
        delta_y = float(target.location.y-location.y)
        transform = self.hero_actor.get_transform()
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        local = [delta_x*forward.x+delta_y*forward.y,
                 -(delta_x*right.x+delta_y*right.y)]
        return local, bool(np.isfinite(local).all()), option

    def _ego_status(self, current_speed, navigation_option=None):
        transform = self.hero_actor.get_transform()
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        acceleration = self.hero_actor.get_acceleration()
        # Training annotations provide scalar speed and set lateral velocity to
        # zero.  Match that contract online instead of injecting an unseen
        # lateral-velocity distribution during turns or slips.
        velocity_local = (
            [float(current_speed)]
            if self._velocity_mode == "scalar_speed"
            else [float(current_speed), 0.0]
        )
        acceleration_local = [
            acceleration.x * forward.x + acceleration.y * forward.y,
            -(acceleration.x * right.x + acceleration.y * right.y),
        ]
        # The model consumes the latest history entry; its current pose is zero.
        command = (road_option_to_one_hot(navigation_option, self._command_num)
                   if navigation_option is not None else self._nearest_command())
        return [0.0, 0.0, 0.0] + velocity_local + acceleration_local + command

    def _request_inference(self, current_jpeg, timestamp, current_speed):
        self._connect()
        target_time = timestamp - self._history_seconds
        previous_jpeg = min(
            self._frames, key=lambda item: abs(item[0] - target_time)
        )[1]
        target_point, target_point_valid, navigation_option = self._navigation_target()
        request = {
            "version": PROTOCOL_VERSION,
            "op": "infer",
            "timestamp": float(timestamp),
            "previous_jpegs": previous_jpeg,
            "current_jpegs": current_jpeg,
            "camera_order": list(CAMERA_ORDER),
            "ego_status": self._ego_status(current_speed, navigation_option),
            "target_point": target_point,
            "target_point_valid": target_point_valid,
        }
        try:
            send_message(self._socket, request)
            response = receive_message(self._socket)
        except Exception:
            self._socket.close()
            self._socket = None
            raise
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Drive-JEPA inference failed"))

        trajectory = np.asarray(response["trajectory"], dtype=np.float32)
        transform = self.hero_actor.get_transform()
        self._world_trajectory = trajectory_to_world(
            trajectory,
            transform.location,
            transform.get_forward_vector(),
            transform.get_right_vector(),
        )
        self._last_diagnostics = response.get("diagnostics", {})
        self._last_inference_time = float(timestamp)

    def run_step(self, input_data, timestamp):
        self._step += 1
        frames = [input_data[name][0] for name in CAMERA_ORDER]
        if len(set(frames)) != 1:
            raise ValueError('Four cameras must come from the same simulator frame')
        current_jpeg = {name: self._camera_jpeg(input_data[name][1], self._jpeg_quality)
                        for name in CAMERA_ORDER}
        self._frames.append((float(timestamp), current_jpeg))
        speed_data = input_data.get("speed", (None, {"speed": 0.0}))[1]
        current_speed = float(speed_data.get("speed", 0.0))

        due = self._last_inference_time is None or (
            float(timestamp) - self._last_inference_time
            >= self._inference_interval - 1e-4
        )
        if due:
            self._request_inference(current_jpeg, float(timestamp), current_speed)

        if self._world_trajectory is None:
            control = carla.VehicleControl()
            control.brake = 1.0
            return control

        transform = self.hero_actor.get_transform()
        local_trajectory = world_to_local_trajectory(
            self._world_trajectory,
            transform.location,
            transform.get_forward_vector(),
            transform.get_right_vector(),
        )
        result = trajectory_to_control(
            local_trajectory,
            current_speed,
            self._controller_state,
            self.config.get("controller", {}),
        )
        self._controller_state = result["state"]

        control = carla.VehicleControl()
        control.steer = result["steer"]
        control.throttle = result["throttle"]
        control.brake = result["brake"]
        control.hand_brake = False

        if (
            self._diagnostic_interval_steps > 0
            and self._step % self._diagnostic_interval_steps == 0
        ):
            location = self.hero_actor.get_location()
            first_point = local_trajectory[0]
            last_point = local_trajectory[-1]
            print(
                "Drive-JEPA control step={} t={:.2f} loc=({:.2f},{:.2f}) "
                "speed={:.2f} target={:.2f} throttle={:.2f} brake={:.2f} steer={:.2f} "
                "trajectory_first=({:.2f},{:.2f}) trajectory_last=({:.2f},{:.2f}) "
                "inference_latency={:.3f} score={} candidate_speed_max={} "
                "best_moving_score={} selected_metrics={} "
                "best_moving_metrics={}".format(
                    self._step,
                    float(timestamp),
                    float(location.x),
                    float(location.y),
                    current_speed,
                    result["target_speed"],
                    control.throttle,
                    control.brake,
                    control.steer,
                    float(first_point[0]),
                    float(first_point[1]),
                    float(last_point[0]),
                    float(last_point[1]),
                    float(self._last_diagnostics.get("latency_seconds", 0.0)),
                    self._last_diagnostics.get("selected_score", "n/a"),
                    self._last_diagnostics.get("candidate_speed_max", "n/a"),
                    self._last_diagnostics.get("best_moving_score", "n/a"),
                    self._last_diagnostics.get("selected_metrics", "n/a"),
                    self._last_diagnostics.get("best_moving_metrics", "n/a"),
                ),
                flush=True,
            )
        return control

    def destroy(self):
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
