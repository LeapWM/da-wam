#!/usr/bin/env python3
"""Serve this repository's Drive-JEPA model to a Python 3.8 CARLA agent."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import cv2
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch
from camera_contract import CAMERA_ORDER

from bench2drive.bridge_protocol import (
    PROTOCOL_VERSION,
    receive_message,
    send_message,
)


REPOSITORY_ROOT = Path('/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/Drive-JEPA-V2-EMA-JQTF')


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    config["_config_path"] = str(Path(path).resolve())
    return config


def resolve_model_settings(config):
    model = dict(config.get("model", {}))
    overrides = {
        "hydra_config": os.environ.get("DRIVE_JEPA_HYDRA_CONFIG"),
        "checkpoint": os.environ.get("DRIVE_JEPA_CHECKPOINT"),
        "device": os.environ.get("DRIVE_JEPA_DEVICE"),
        "name": os.environ.get("DRIVE_JEPA_MODEL_NAME"),
    }
    model.update({key: value for key, value in overrides.items() if value})

    model_root = Path(os.environ.get("DRIVE_JEPA_ROOT", REPOSITORY_ROOT)).resolve()
    required = ("hydra_config", "checkpoint")
    missing = [name for name in required if not model.get(name)]
    if missing:
        raise ValueError("model config is missing: {}".format(", ".join(missing)))

    paths = {
        "model_root": model_root,
        "hydra_config": Path(model["hydra_config"]).expanduser().resolve(),
        "checkpoint": Path(model["checkpoint"]).expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError("{} does not exist: {}".format(name, path))
    navsim_root = model_root / "navsim_v1"
    if not (navsim_root / "navsim").is_dir():
        raise FileNotFoundError("NAVSIM package not found under {}".format(navsim_root))
    return model, navsim_root, paths


def _navsim_front_lidar_to_image():
    """NAVSIM CAM_F0 calibration followed by the training-time 0.4 resize."""
    sensor_to_lidar_rotation = np.asarray(
        [
            [0.00309581, -0.02527090, 0.99967585],
            [-0.99995684, -0.00883471, 0.00287334],
            [0.00875923, -0.99964160, -0.02529716],
        ],
        dtype=np.float32,
    )
    sensor_to_lidar_translation = np.asarray(
        [1.6701001, -0.02587495, 1.5226235], dtype=np.float32
    )
    intrinsic = np.asarray(
        [[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    lidar_to_camera_rotation = np.linalg.inv(sensor_to_lidar_rotation)
    lidar_to_camera_translation = (
        sensor_to_lidar_translation @ lidar_to_camera_rotation.T
    )
    lidar_to_camera = np.eye(4, dtype=np.float32)
    lidar_to_camera[:3, :3] = lidar_to_camera_rotation.T
    lidar_to_camera[3, :3] = -lidar_to_camera_translation
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:3, :3] = intrinsic
    lidar_to_image = viewpad @ lidar_to_camera.T
    lidar_to_image[:2] *= 0.4
    return lidar_to_image


def _bench2drive_front_lidar_to_image():
    """Return the fixed Base CAM_FRONT projection used by the training cache."""
    return np.asarray(
        [
            [365.605889726, 256.0, 0.0, -304.64],
            [0.0, 136.492890995, -346.545867039, -254.090439369],
            [0.0, 1.0, 0.0, -1.19],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _front_lidar_to_image(profile):
    if profile == "navsim":
        return _navsim_front_lidar_to_image()
    if profile == "bench2drive":
        return _bench2drive_front_lidar_to_image()
    raise ValueError("unknown feature profile: {}".format(profile))


def _decode_camera(jpeg):
    bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode JPEG camera frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] <= 56:
        raise ValueError("camera frame is too short for the NAVSIM crop")
    cropped = rgb[28:-28]
    resized = cv2.resize(cropped, (512, 256), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(resized.transpose(2, 0, 1))
    return torch.from_numpy(tensor).float() / 255.0


def _normalize_ego_status(values, expected_dim):
    """Adapt the four-command bridge status to a six-command B2D model."""
    ego_status = np.asarray(values, dtype=np.float32)
    if ego_status.shape == (expected_dim,) and np.isfinite(ego_status).all():
        return ego_status
    if (
        ego_status.shape == (11,)
        and expected_dim == 13
        and np.isfinite(ego_status).all()
    ):
        # Both formats share seven kinematic columns followed by commands.
        # The bridge emits the four NAVSIM commands; Bench2Drive adds two
        # lane-change commands at the end, which are inactive here.
        return np.pad(ego_status, (0, 2), mode="constant")
    raise ValueError(
        "ego_status must contain {} finite values (or 11 values for the "
        "Bench2Drive compatibility mapping)".format(expected_dim)
    )


def _normalize_target_point(request):
    target = np.asarray(request.get("target_point"), dtype=np.float32)
    valid = bool(request.get("target_point_valid", False))
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError("target_point must contain two finite local xy values")
    return target, valid


def build_features(
    request,
    device,
    expected_ego_status_dim=11,
    feature_config=None,
):
    from multiview_inputs import build_online_features
    return build_online_features(request, device, expected_ego_status_dim, feature_config)



class DriveJEPAModelRuntime:
    def __init__(self, config):
        model, navsim_root, paths = resolve_model_settings(config)
        self.device = torch.device(model.get("device", "cuda:0"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the Drive-JEPA process")

        sys.path.insert(0, str(navsim_root / "vjepa2"))
        sys.path.insert(0, str(navsim_root))
        os.chdir(str(navsim_root))
        os.environ.setdefault("NAVSIM_DEVKIT_ROOT", str(navsim_root))
        os.environ.setdefault("NAVSIM_EXP_ROOT", "/mnt/c2-worldmodel/2639639/navsim_exp")
        os.environ.setdefault(
            "OPENSCENE_DATA_ROOT", "/mnt/c2-worldmodel/training_data/OpenScene/dataset"
        )
        os.environ.setdefault("NUPLAN_DATA_ROOT", os.environ["OPENSCENE_DATA_ROOT"])
        os.environ.setdefault(
            "NUPLAN_MAPS_ROOT", os.environ["OPENSCENE_DATA_ROOT"] + "/maps"
        )

        print("Loading Drive-JEPA model from {}".format(paths["model_root"]), flush=True)
        print("Loading checkpoint {}".format(paths["checkpoint"]), flush=True)
        hydra_config = OmegaConf.load(str(paths["hydra_config"]))
        hydra_config.agent.checkpoint_path = str(paths["checkpoint"])
        hydra_config.agent.cache_data = False
        ego_status_config = dict(config.get("ego_status", {}))
        velocity_mode = str(
            ego_status_config.get("velocity_mode", "longitudinal_lateral")
        )
        if velocity_mode == "scalar_speed":
            kinematic_dim = 6
        elif velocity_mode == "longitudinal_lateral":
            kinematic_dim = 7
        else:
            raise ValueError("unsupported ego_status.velocity_mode")
        self.expected_ego_status_dim = kinematic_dim + int(
            getattr(hydra_config.agent.config, "command_num", 4)
        )
        self.feature_config = dict(config.get("features", {}))
        self.agent = instantiate(hydra_config.agent)
        self.agent.initialize()
        self.agent.to(self.device)
        self.agent.eval()
        self.use_bf16 = bool(model.get("use_bf16", True)) and self.device.type == "cuda"
        self.model_name = model.get("name", paths["checkpoint"].name)
        device_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else "CPU"
        )
        print(
            "Drive-JEPA ready on {} ({})".format(self.device, device_name), flush=True
        )
        print(
            "Expected ego-status dimension: {}".format(
                self.expected_ego_status_dim
            ),
            flush=True,
        )

    def infer(self, request):
        started = time.perf_counter()
        features = build_features(
            request,
            self.device,
            self.expected_ego_status_dim,
            self.feature_config,
        )
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_bf16,
            ):
                output = self.agent(features)
        trajectory = output["trajectory"][0].float().cpu().numpy()
        diagnostics = {"latency_seconds": time.perf_counter() - started}
        if "selected_index" in output:
            diagnostics["selected_index"] = int(output["selected_index"][0].item())
        if "selection_score" in output and "selected_index" in output:
            index = diagnostics["selected_index"]
            diagnostics["selected_score"] = float(
                output["selection_score"][0, index].float().item()
            )
        if "proposals" in output:
            proposals = output["proposals"][0, :, :, :2].float()
            origins = torch.zeros(
                (proposals.shape[0], 1, 2),
                device=proposals.device,
                dtype=proposals.dtype,
            )
            segments = torch.linalg.vector_norm(
                torch.diff(torch.cat([origins, proposals], dim=1), dim=1),
                dim=-1,
            )
            speed_points = min(4, segments.shape[1])
            candidate_speeds = torch.median(
                segments[:, :speed_points], dim=1
            ).values / 0.5
            candidate_progress = segments.sum(dim=1)
            diagnostics.update(
                {
                    "candidate_speed_median": float(
                        candidate_speeds.median().item()
                    ),
                    "candidate_speed_max": float(candidate_speeds.max().item()),
                    "candidate_progress_median": float(
                        candidate_progress.median().item()
                    ),
                    "candidate_progress_max": float(
                        candidate_progress.max().item()
                    ),
                }
            )
            if "selected_index" in diagnostics:
                index = diagnostics["selected_index"]
                diagnostics["selected_target_speed"] = float(
                    candidate_speeds[index].item()
                )
                diagnostics["selected_progress"] = float(
                    candidate_progress[index].item()
                )
                if "pred_metric_scores" in output:
                    diagnostics["selected_metrics"] = [
                        float(value)
                        for value in output["pred_metric_scores"][0, index]
                        .float()
                        .tolist()
                    ]
                if "pred_ep" in output:
                    diagnostics["selected_ep"] = float(
                        output["pred_ep"][0, index].float().item()
                    )
            if "selection_score" in output:
                scores = output["selection_score"][0].float()
                moving = candidate_speeds >= 0.5
                if bool(moving.any()):
                    moving_scores = scores.masked_fill(~moving, float("-inf"))
                    moving_index = int(torch.argmax(moving_scores).item())
                    diagnostics["best_moving_index"] = moving_index
                    diagnostics["best_moving_score"] = float(
                        scores[moving_index].item()
                    )
                    diagnostics["best_moving_speed"] = float(
                        candidate_speeds[moving_index].item()
                    )
                    diagnostics["best_moving_progress"] = float(
                        candidate_progress[moving_index].item()
                    )
                    if "pred_metric_scores" in output:
                        diagnostics["best_moving_metrics"] = [
                            float(value)
                            for value in output["pred_metric_scores"][
                                0, moving_index
                            ]
                            .float()
                            .tolist()
                        ]
                    if "pred_ep" in output:
                        diagnostics["best_moving_ep"] = float(
                            output["pred_ep"][0, moving_index].float().item()
                        )
        return trajectory.tolist(), diagnostics

    @staticmethod
    def synthetic_request(expected_ego_status_dim=11, include_target_point=False):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image[:, :, 1] = 96
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("synthetic JPEG encoding failed")
        ego_status = [0.0] * int(expected_ego_status_dim)
        if expected_ego_status_dim == 11:
            ego_status[8] = 1.0
        elif expected_ego_status_dim == 12:
            ego_status[9] = 1.0
        elif expected_ego_status_dim == 13:
            ego_status[10] = 1.0
        else:
            raise ValueError("unsupported synthetic ego-status dimension")
        request = {
            "current_jpegs": {name: encoded.tobytes() for name in CAMERA_ORDER},
            "previous_jpegs": {name: encoded.tobytes() for name in CAMERA_ORDER},
            "camera_order": list(CAMERA_ORDER),
            "ego_status": ego_status,
        }
        if include_target_point:
            request.update(target_point=[5.0, 0.0], target_point_valid=True)
        return request


def serve(config):
    runtime = DriveJEPAModelRuntime(config)
    if config.get("model", {}).get("warmup", True):
        trajectory, diagnostics = runtime.infer(
            runtime.synthetic_request(runtime.expected_ego_status_dim,
                                      bool(runtime.feature_config.get("target_point", False)))
        )
        print(
            "Warmup passed: {} poses, {:.3f}s".format(
                len(trajectory), diagnostics["latency_seconds"]
            ),
            flush=True,
        )

    bridge = config["bridge"]
    host = bridge.get("host", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("the pickle bridge must remain bound to loopback")
    port = int(
        os.environ.get("DRIVE_JEPA_BRIDGE_PORT", bridge.get("port", 50123))
    )
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(2)
    print("Drive-JEPA bridge listening on {}:{}".format(host, port), flush=True)

    while True:
        connection, address = server.accept()
        connection.settimeout(float(bridge.get("timeout_seconds", 60.0)))
        print("Bridge client connected: {}".format(address), flush=True)
        try:
            while True:
                request = receive_message(connection)
                if request.get("version") != PROTOCOL_VERSION:
                    raise ValueError("unsupported bridge protocol version")
                if request.get("op") == "ping":
                    send_message(
                        connection,
                        {
                            "ok": True,
                            "version": PROTOCOL_VERSION,
                            "model": runtime.model_name,
                        },
                    )
                    continue
                if request.get("op") != "infer":
                    raise ValueError("unknown bridge operation")
                try:
                    trajectory, diagnostics = runtime.infer(request)
                    send_message(
                        connection,
                        {
                            "ok": True,
                            "trajectory": trajectory,
                            "diagnostics": diagnostics,
                        },
                    )
                except Exception as error:
                    traceback.print_exc()
                    send_message(connection, {"ok": False, "error": repr(error)})
        except (EOFError, ConnectionError, socket.timeout):
            pass
        finally:
            connection.close()
            print("Bridge client disconnected", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    model, navsim_root, paths = resolve_model_settings(config)
    if args.dry_run:
        print("configuration valid")
        print("model root : {}".format(paths["model_root"]))
        print("NAVSIM root: {}".format(navsim_root))
        print("hydra config: {}".format(paths["hydra_config"]))
        print("checkpoint : {}".format(paths["checkpoint"]))
        print("device     : {}".format(model.get("device", "cuda:0")))
        return
    if args.self_test:
        runtime = DriveJEPAModelRuntime(config)
        trajectory, diagnostics = runtime.infer(
                runtime.synthetic_request(runtime.expected_ego_status_dim,
                                          bool(runtime.feature_config.get("target_point", False)))
        )
        print(
            "self-test trajectory shape: ({}, {})".format(
                len(trajectory), len(trajectory[0])
            )
        )
        print("self-test diagnostics: {}".format(diagnostics))
        return
    serve(config)


if __name__ == "__main__":
    main()
