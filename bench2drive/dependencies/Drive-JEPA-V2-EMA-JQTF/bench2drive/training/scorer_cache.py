"""Build Drive-JEPA scorer supervision directly from Bench2Drive archives.

The Base release stores each clip as a ``.tar.gz`` archive and each frame
annotation as a gzipped JSON member.  This module deliberately reads archives
sequentially and never extracts a whole clip.  The resulting cache contains
the information that cannot be reconstructed from model features alone:

* the expert future ego trajectory in the current ego frame;
* future vehicle/walker boxes in the same frame;
* the expert progress distance used by the Formula-progress head.

Drivable-area supervision is evaluated later against the official HD-map
cache because it depends on the model's generated candidate trajectories.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


_ANNO_RE = re.compile(r"(?:^|/)anno/(?P<frame>\d+)\.json\.gz$")
_DYNAMIC_CLASSES = frozenset({"vehicle", "walker"})
_LEFT_TO_RIGHT = np.diag([1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class ScorerCacheConfig:
    """Sampling and padding policy for a scorer cache."""

    horizon: int = 8
    frame_stride: int = 5
    sample_stride: int = 5
    max_agents: int = 64
    min_frame: int = 0

    def __post_init__(self) -> None:
        for name in ("horizon", "frame_stride", "sample_stride", "max_agents"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_frame < 0:
            raise ValueError("min_frame must be non-negative")


def _read_annotations(archive_path: Path) -> Dict[int, Mapping[str, object]]:
    annotations: Dict[int, Mapping[str, object]] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            match = _ANNO_RE.search(member.name)
            if match is None or not member.isfile():
                continue
            file_obj = archive.extractfile(member)
            if file_obj is None:
                raise RuntimeError(f"could not read {member.name} from {archive_path}")
            try:
                payload = gzip.decompress(file_obj.read())
                annotations[int(match.group("frame"))] = json.loads(payload)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid annotation {member.name} in {archive_path}: {error}"
                ) from error
    if not annotations:
        raise ValueError(f"archive contains no anno/*.json.gz members: {archive_path}")
    return annotations


def _ego_box(annotation: Mapping[str, object]) -> Mapping[str, object]:
    for box in annotation.get("bounding_boxes", []):
        if box.get("class") == "ego_vehicle":
            return box
    raise ValueError("annotation does not contain an ego_vehicle bounding box")


def _world2ego(annotation: Mapping[str, object]) -> np.ndarray:
    raw_matrix = np.asarray(_ego_box(annotation)["world2ego"], dtype=np.float64)
    matrix = _LEFT_TO_RIGHT @ raw_matrix @ _LEFT_TO_RIGHT
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid world2ego matrix with shape {matrix.shape}")
    return matrix


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    right_handed = points[..., :3].copy()
    right_handed[..., 1] *= -1.0
    homogeneous = np.concatenate(
        [right_handed, np.ones((*points.shape[:-1], 1), dtype=np.float64)],
        axis=-1,
    )
    return np.einsum("ij,...j->...i", matrix, homogeneous)[..., :3]


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _future_ego_pose(
    current_world2ego: np.ndarray,
    current_annotation: Mapping[str, object],
    future_annotation: Mapping[str, object],
) -> np.ndarray:
    future_ego = _ego_box(future_annotation)
    location = np.asarray(future_ego["location"], dtype=np.float64)
    xy = _transform_points(current_world2ego, location)[..., :2]
    future_world2ego = _world2ego(future_annotation)
    future_ego2world = np.linalg.inv(future_world2ego)
    future_to_current = current_world2ego @ future_ego2world
    relative_yaw = math.atan2(future_to_current[1, 0], future_to_current[0, 0])
    return np.asarray([xy[0], xy[1], _wrap_angle(relative_yaw)], dtype=np.float32)


def _box_corners_world(box: Mapping[str, object]) -> Optional[np.ndarray]:
    transform_key = None
    if "world2vehicle" in box:
        transform_key = "world2vehicle"
    elif "world2ped" in box:
        transform_key = "world2ped"
    if transform_key is not None and "center" in box and "extent" in box:
        world2actor = np.asarray(box[transform_key], dtype=np.float64)
        center = np.asarray(box["center"], dtype=np.float64)
        extent = np.asarray(box["extent"], dtype=np.float64)
        if (
            world2actor.shape == (4, 4)
            and center.shape == (3,)
            and extent.shape == (3,)
            and np.isfinite(world2actor).all()
            and np.isfinite(center).all()
            and np.isfinite(extent).all()
        ):
            actor2world = np.linalg.inv(world2actor)
            local = np.asarray(
                [
                    [extent[0], extent[1]],
                    [-extent[0], extent[1]],
                    [-extent[0], -extent[1]],
                    [extent[0], -extent[1]],
                ],
                dtype=np.float64,
            )
            world_xy = local @ actor2world[:2, :2].T + center[:2]
            return np.concatenate(
                [world_xy, np.full((4, 1), center[2] - extent[2])], axis=-1
            )

    corners = np.asarray(box.get("world_cord", []), dtype=np.float64)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        return None

    # CARLA emits each footprint vertex twice (bottom/top) in adjacent pairs.
    # Pitch/roll make their XY values slightly different, so numeric
    # deduplication incorrectly produces more than four points.
    unique_xy = corners[[0, 2, 4, 6], :2]
    center = unique_xy.mean(axis=0)
    order = np.argsort(np.arctan2(unique_xy[:, 1] - center[1], unique_xy[:, 0] - center[0]))
    ordered_xy = unique_xy[order]
    z = np.full((4, 1), float(corners[:, 2].min()), dtype=np.float64)
    return np.concatenate([ordered_xy, z], axis=-1)


def _actor_id(box: Mapping[str, object]) -> str:
    return f"{box.get('class', 'unknown')}:{box.get('id', '')}"


def _actors_by_id(annotation: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    actors: Dict[str, Mapping[str, object]] = {}
    for box in annotation.get("bounding_boxes", []):
        if box.get("class") not in _DYNAMIC_CLASSES:
            continue
        if _box_corners_world(box) is None:
            continue
        actors[_actor_id(box)] = box
    return actors


def _select_actor_ids(
    current_annotation: Mapping[str, object],
    future_annotations: Sequence[Mapping[str, object]],
    current_world2ego: np.ndarray,
    max_agents: int,
) -> List[str]:
    first_seen: Dict[str, Tuple[int, float]] = {}
    for time_index, annotation in enumerate(future_annotations):
        for actor_id, box in _actors_by_id(annotation).items():
            corners = _box_corners_world(box)
            assert corners is not None
            local_center = _transform_points(current_world2ego, corners).mean(axis=0)
            distance = float(np.linalg.norm(local_center[:2]))
            previous = first_seen.get(actor_id)
            candidate = (time_index, distance)
            if previous is None or candidate < previous:
                first_seen[actor_id] = candidate
    return [actor_id for actor_id, _ in sorted(first_seen.items(), key=lambda item: item[1])[:max_agents]]


def _build_future_boxes(
    current_annotation: Mapping[str, object],
    future_annotations: Sequence[Mapping[str, object]],
    max_agents: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_world2ego = _world2ego(current_annotation)
    actor_ids = _select_actor_ids(
        current_annotation,
        future_annotations,
        current_world2ego,
        max_agents,
    )
    id_to_index = {actor_id: index for index, actor_id in enumerate(actor_ids)}
    horizon = len(future_annotations)
    boxes = np.zeros((max_agents, horizon, 4, 2), dtype=np.float32)
    valid = np.zeros((max_agents, horizon), dtype=np.bool_)
    actor_types = np.zeros((max_agents,), dtype=np.int8)

    for time_index, annotation in enumerate(future_annotations):
        for actor_id, box in _actors_by_id(annotation).items():
            actor_index = id_to_index.get(actor_id)
            if actor_index is None:
                continue
            world_corners = _box_corners_world(box)
            assert world_corners is not None
            local_corners = _transform_points(current_world2ego, world_corners)[..., :2]
            boxes[actor_index, time_index] = local_corners.astype(np.float32)
            valid[actor_index, time_index] = True
            actor_types[actor_index] = 1 if box.get("class") == "vehicle" else 2
    return boxes, valid, actor_types


def _build_sample(
    clip_name: str,
    frame_id: int,
    annotations: Mapping[int, Mapping[str, object]],
    config: ScorerCacheConfig,
) -> Mapping[str, object]:
    future_ids = [frame_id + config.frame_stride * index for index in range(1, config.horizon + 1)]
    future_annotations = [annotations[index] for index in future_ids]
    current = annotations[frame_id]
    world2ego = _world2ego(current)
    trajectory = np.stack(
        [_future_ego_pose(world2ego, current, annotation) for annotation in future_annotations],
        axis=0,
    )
    future_boxes, future_box_valid, actor_types = _build_future_boxes(
        current,
        future_annotations,
        config.max_agents,
    )
    progress = float(np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 2)), trajectory[:, :2]], axis=0), axis=0), axis=-1).sum())
    command = int(current.get("command_near", current.get("next_command", 0)))
    town_match = re.search(r"(Town\d+(?:HD)?)", clip_name)
    town_name = town_match.group(1) if town_match else ""
    token = f"{clip_name}/{frame_id:05d}"
    return {
        "token": token,
        "clip": clip_name,
        "frame_id": int(frame_id),
        "town_name": town_name,
        "command": command,
        "trajectory": trajectory,
        "future_box_corners": future_boxes,
        "future_box_valid": future_box_valid,
        "actor_types": actor_types,
        "world2ego": world2ego.astype(np.float32),
        "expert_progress": np.float32(progress),
    }


def build_clip_scorer_cache(
    archive_path: Path,
    config: ScorerCacheConfig = ScorerCacheConfig(),
) -> Dict[str, Mapping[str, object]]:
    """Return scorer samples from one Bench2Drive clip archive."""

    archive_path = Path(archive_path)
    annotations = _read_annotations(archive_path)
    clip_name = archive_path.name
    if clip_name.endswith(".tar.gz"):
        clip_name = clip_name[: -len(".tar.gz")]
    available = set(annotations)
    samples: Dict[str, Mapping[str, object]] = {}
    for frame_id in sorted(available):
        if frame_id < config.min_frame or frame_id % config.sample_stride != 0:
            continue
        future_ids = [frame_id + config.frame_stride * index for index in range(1, config.horizon + 1)]
        if not all(index in available for index in future_ids):
            continue
        sample = _build_sample(clip_name, frame_id, annotations, config)
        samples[str(sample["token"])] = sample
    return samples


def _dump_cache(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as file_obj:
        pickle.dump(dict(payload), file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def build_scorer_cache(
    archives: Iterable[Path],
    output_path: Path,
    config: ScorerCacheConfig = ScorerCacheConfig(),
) -> Dict[str, object]:
    """Build and atomically write a multi-clip scorer cache."""

    samples: MutableMapping[str, Mapping[str, object]] = {}
    clip_count = 0
    for archive_path in archives:
        clip_samples = build_clip_scorer_cache(Path(archive_path), config)
        overlap = samples.keys() & clip_samples.keys()
        if overlap:
            raise ValueError(f"duplicate scorer tokens: {sorted(overlap)[:3]}")
        samples.update(clip_samples)
        clip_count += 1
    payload: Dict[str, object] = {
        "version": 1,
        "config": {
            "horizon": config.horizon,
            "frame_stride": config.frame_stride,
            "sample_stride": config.sample_stride,
            "max_agents": config.max_agents,
            "min_frame": config.min_frame,
        },
        "clip_count": clip_count,
        "samples": dict(samples),
    }
    _dump_cache(Path(output_path), payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=1)
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--max-agents", type=int, default=64)
    parser.add_argument("--min-frame", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    archives = sorted(args.data_root.glob("*.tar.gz"))
    if not 0 < args.val_percent < 100:
        raise ValueError("--val-percent must be within (0, 100)")
    if args.split != "all":
        def is_validation_clip(path):
            digest = hashlib.sha1(path.name.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], byteorder="big") % 100
            return bucket < args.val_percent

        want_validation = args.split == "val"
        archives = [path for path in archives if is_validation_clip(path) == want_validation]
    if args.max_clips < 0:
        raise ValueError("--max-clips must be non-negative")
    if args.max_clips:
        archives = archives[: args.max_clips]
    if not archives:
        raise FileNotFoundError(f"no .tar.gz clips found under {args.data_root}")
    config = ScorerCacheConfig(
        horizon=args.horizon,
        frame_stride=args.frame_stride,
        sample_stride=args.sample_stride,
        max_agents=args.max_agents,
        min_frame=args.min_frame,
    )
    payload = build_scorer_cache(archives, args.output, config)
    print(
        f"wrote {len(payload['samples'])} samples from "
        f"{payload['clip_count']} clips to {args.output}"
    )


if __name__ == "__main__":
    main()
