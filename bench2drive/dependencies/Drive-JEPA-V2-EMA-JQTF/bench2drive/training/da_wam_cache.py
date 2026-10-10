"""Build DA-WAM training caches from compressed Bench2Drive clips.

The generated directory follows NAVSIM's ``CacheOnlyDataset`` layout.  The
source archives are streamed and are never extracted wholesale.  Camera
tensors are stored as uint8 to keep the cache compact; DriveJEPAModel accepts
both these tensors and the legacy float [0, 1] NAVSIM tensors.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import re
import tarfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import cv2
import numpy as np
import torch

from bench2drive.training.scorer_cache import (
    ScorerCacheConfig,
    build_scorer_cache,
)


_ANNO_RE = re.compile(r"(?:^|/)anno/(?P<frame>\d+)\.json\.gz$")
_FRONT_RE = re.compile(r"(?:^|/)camera/rgb_front/(?P<frame>\d+)\.jpg$")
_LEFT_TO_RIGHT = np.diag([1.0, -1.0, 1.0, 1.0])
_STANDARD_TO_UE4 = np.asarray(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_LIDAR_TO_RIGHT_HANDED_EGO = np.asarray(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def is_validation_clip(path: Path, val_percent: int = 10) -> bool:
    """Return the deterministic scorer-cache train/validation assignment."""

    digest = hashlib.sha1(Path(path).name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big") % 100 < val_percent


def select_archives(
    data_root: Path,
    split: str,
    val_percent: int,
    max_clips: int,
    towns: Sequence[str] = (),
) -> List[Path]:
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    if not 0 < val_percent < 100:
        raise ValueError("val_percent must be within (0, 100)")
    if max_clips < 0:
        raise ValueError("max_clips must be non-negative")
    want_validation = split == "val"
    town_filters = tuple(towns)
    archives = [
        path
        for path in sorted(Path(data_root).glob("*.tar.gz"))
        if is_validation_clip(path, val_percent) == want_validation
        and (
            not town_filters
            or any(f"_{town}_" in path.name for town in town_filters)
        )
    ]
    return archives[:max_clips] if max_clips else archives


def preprocess_front_jpeg(payload: bytes) -> torch.Tensor:
    """Decode a B2D front JPEG using the deployed bridge preprocessing."""

    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode Bench2Drive front JPEG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] <= 56:
        raise ValueError(f"front image is too short for crop: {rgb.shape}")
    rgb = rgb[28:-28]
    rgb = cv2.resize(rgb, (512, 256), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def build_ego_status(
    annotation: Mapping[str, object],
    command_num: int = 6,
    velocity_mode: str = "longitudinal_lateral",
) -> torch.Tensor:
    """Build [pose, velocity, acceleration, command] in B2D right-hand axes."""

    command = int(annotation.get("command_near", 4))
    if command < 1 or command > command_num:
        command = 4
    command_one_hot = np.zeros(command_num, dtype=np.float32)
    command_one_hot[command - 1] = 1.0
    acceleration = np.asarray(annotation.get("acceleration", [0.0, 0.0]), dtype=np.float32)
    if acceleration.size < 2:
        raise ValueError("annotation acceleration must contain x and y")
    if velocity_mode == "scalar_speed":
        velocity = np.asarray([float(annotation.get("speed", 0.0))], dtype=np.float32)
    elif velocity_mode == "longitudinal_lateral":
        velocity = np.asarray(
            [float(annotation.get("speed", 0.0)), 0.0], dtype=np.float32
        )
    else:
        raise ValueError(f"unsupported velocity_mode: {velocity_mode}")
    status = np.concatenate(
        [
            np.zeros(3, dtype=np.float32),
            velocity,
            np.asarray([acceleration[0], -acceleration[1]], dtype=np.float32),
            command_one_hot,
        ]
    )
    return torch.from_numpy(status[None])


def front_lidar_to_processed_image(annotation: Mapping[str, object]) -> torch.Tensor:
    """Project the official right-handed B2D lidar frame into the cached image."""

    sensors = annotation["sensors"]
    camera = sensors["CAM_FRONT"]
    lidar = sensors["LIDAR_TOP"]
    cam2ego = (
        _LEFT_TO_RIGHT
        @ np.asarray(camera["cam2ego"], dtype=np.float64)
        @ _STANDARD_TO_UE4
    )
    lidar2ego = (
        _LEFT_TO_RIGHT
        @ np.asarray(lidar["lidar2ego"], dtype=np.float64)
        @ _LEFT_TO_RIGHT
        @ _LIDAR_TO_RIGHT_HANDED_EGO
    )
    lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[:3, :3] = np.asarray(camera["intrinsic"], dtype=np.float64)
    projection = intrinsic @ lidar2cam

    width = float(camera.get("image_size_x", 1600))
    height = float(camera.get("image_size_y", 900))
    crop_height = height - 56.0
    image_transform = np.eye(4, dtype=np.float64)
    image_transform[0, 0] = 512.0 / width
    image_transform[1, 1] = 256.0 / crop_height
    image_transform[1, 3] = -28.0 * image_transform[1, 1]
    projection = (image_transform @ projection).astype(np.float32)
    if not np.isfinite(projection).all():
        raise ValueError("non-finite B2D front lidar2img projection")
    return torch.from_numpy(np.repeat(projection[None], 4, axis=0))


def _read_required_members(
    archive_path: Path,
    annotation_frames: Set[int],
    image_frames: Set[int],
) -> Tuple[Dict[int, Mapping[str, object]], Dict[int, torch.Tensor]]:
    annotations: Dict[int, Mapping[str, object]] = {}
    images: Dict[int, torch.Tensor] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            anno_match = _ANNO_RE.search(member.name)
            image_match = _FRONT_RE.search(member.name)
            if anno_match is not None:
                frame = int(anno_match.group("frame"))
                if frame not in annotation_frames:
                    continue
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    raise RuntimeError(f"could not read {member.name}")
                annotations[frame] = json.loads(gzip.decompress(file_obj.read()))
            elif image_match is not None:
                frame = int(image_match.group("frame"))
                if frame not in image_frames:
                    continue
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    raise RuntimeError(f"could not read {member.name}")
                images[frame] = preprocess_front_jpeg(file_obj.read())
    missing_annotations = annotation_frames - annotations.keys()
    missing_images = image_frames - images.keys()
    if missing_annotations or missing_images:
        raise ValueError(
            f"{archive_path.name} is missing required members: "
            f"annotations={sorted(missing_annotations)}, images={sorted(missing_images)}"
        )
    return annotations, images


def _dump_pickle(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as file_obj:
        pickle.dump(dict(payload), file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _token_directory_name(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", token)


def build_clip_da_wam_cache(
    archive_path: Path,
    scorer_samples: Mapping[str, Mapping[str, object]],
    cache_root: Path,
    split: str,
    history_stride: int = 5,
    future_stride: int = 5,
    command_num: int = 6,
    force: bool = False,
) -> int:
    """Write all aligned Drive-JEPA samples for one archive."""

    clip_name = archive_path.name[: -len(".tar.gz")]
    clip_samples = [
        sample
        for sample in scorer_samples.values()
        if sample["clip"] == clip_name and int(sample["frame_id"]) >= history_stride
    ]
    clip_samples.sort(key=lambda sample: int(sample["frame_id"]))
    if not clip_samples:
        return 0
    current_frames = {int(sample["frame_id"]) for sample in clip_samples}
    image_frames = set(current_frames)
    image_frames.update(frame - history_stride for frame in current_frames)
    image_frames.update(frame + future_stride for frame in current_frames)
    annotations, images = _read_required_members(
        archive_path, current_frames, image_frames
    )

    written = 0
    for sample in clip_samples:
        frame = int(sample["frame_id"])
        token = str(sample["token"])
        token_root = Path(cache_root) / split / _token_directory_name(token)
        feature_path = token_root / "da_wam_feature.gz"
        target_path = token_root / "da_wam_target.gz"
        if not force and feature_path.is_file() and target_path.is_file():
            written += 1
            continue
        annotation = annotations[frame]
        projection = front_lidar_to_processed_image(annotation)
        image_shapes = torch.tensor(
            [[256.0, 512.0, 3.0]] * 4, dtype=torch.float32
        )
        features = {
            "camera_feature_1": images[frame],
            "camera_feature_2": images[frame - history_stride],
            "ego_status": build_ego_status(annotation, command_num),
            "lidar2img": projection,
            "img_shape": image_shapes,
        }
        targets = {
            "trajectory": torch.as_tensor(sample["trajectory"], dtype=torch.float32),
            "token": token,
            "future_camera_features": images[frame + future_stride].unsqueeze(0),
            "future_camera_offset_order": torch.tensor([1], dtype=torch.int64),
        }
        _dump_pickle(feature_path, features)
        _dump_pickle(target_path, targets)
        written += 1
    return written


def build_da_wam_cache(
    archives: Sequence[Path],
    cache_root: Path,
    scorer_cache_path: Path,
    split: str,
    scorer_config: ScorerCacheConfig,
    history_stride: int = 5,
    future_stride: int = 5,
    command_num: int = 6,
    force: bool = False,
) -> Mapping[str, object]:
    scorer_payload = build_scorer_cache(
        archives, scorer_cache_path, scorer_config
    )
    samples = scorer_payload["samples"]
    sample_count = 0
    for archive_path in archives:
        sample_count += build_clip_da_wam_cache(
            archive_path,
            samples,
            cache_root,
            split,
            history_stride=history_stride,
            future_stride=future_stride,
            command_num=command_num,
            force=force,
        )
        print(f"cached {archive_path.name}")
    manifest = {
        "version": 1,
        "split": split,
        "clip_count": len(archives),
        "sample_count": sample_count,
        "archives": [path.name for path in archives],
        "scorer_cache": str(Path(scorer_cache_path).resolve()),
        "command_num": command_num,
        "history_stride": history_stride,
        "future_stride": future_stride,
        "image_dtype": "uint8",
    }
    manifest_path = Path(cache_root) / f"{split}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--scorer-cache", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--max-clips", type=int, default=1)
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--history-stride", type=int, default=5)
    parser.add_argument("--future-stride", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--max-agents", type=int, default=64)
    parser.add_argument("--command-num", type=int, default=6)
    parser.add_argument(
        "--towns",
        nargs="*",
        default=(),
        help="optional town allow-list, for example --towns Town03 Town04",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    archives = select_archives(
        args.data_root,
        args.split,
        args.val_percent,
        args.max_clips,
        towns=args.towns,
    )
    if not archives:
        raise FileNotFoundError(
            f"no {args.split} .tar.gz clips found under {args.data_root}"
        )
    scorer_config = ScorerCacheConfig(
        horizon=args.horizon,
        frame_stride=args.frame_stride,
        sample_stride=args.sample_stride,
        max_agents=args.max_agents,
        min_frame=args.history_stride,
    )
    manifest = build_da_wam_cache(
        archives,
        args.cache_root,
        args.scorer_cache,
        args.split,
        scorer_config,
        history_stride=args.history_stride,
        future_stride=args.future_stride,
        command_num=args.command_num,
        force=args.force,
    )
    print(
        f"wrote {manifest['sample_count']} {args.split} samples from "
        f"{manifest['clip_count']} clips to {args.cache_root}"
    )


if __name__ == "__main__":
    main()
