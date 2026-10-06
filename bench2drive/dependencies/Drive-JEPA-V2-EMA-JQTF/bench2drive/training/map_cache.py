"""Convert official Bench2Drive HD-map NPZ files to a compact scorer cache."""

from __future__ import annotations

import argparse
import gzip
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


_TOWN_RE = re.compile(r"(Town\d+(?:HD)?)")


def _load_official_map(path: Path) -> Mapping[object, object]:
    # Official Bench2Drive maps contain a nested object array.  Do not use
    # this loader for untrusted NPZ files because NumPy object arrays unpickle.
    with np.load(path, allow_pickle=True) as archive:
        key = "arr" if "arr" in archive.files else archive.files[0]
        raw = archive[key]
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            raw = raw.item()
        else:
            raw = dict(raw.tolist())
    if not isinstance(raw, Mapping):
        raise ValueError(f"unexpected HD-map payload in {path}: {type(raw)!r}")
    return raw


def _point_xyz(point) -> Sequence[float]:
    # Center records are ((x,y,z), (roll,pitch,yaw), junction_flag).
    if not isinstance(point, (tuple, list, np.ndarray)) or len(point) < 1:
        raise ValueError(f"invalid center point: {point!r}")
    location = point[0]
    if not isinstance(location, (tuple, list, np.ndarray)) or len(location) < 2:
        raise ValueError(f"invalid center-point location: {location!r}")
    return location


def compact_map_file(
    path: Path,
    point_stride: int = 10,
    lane_half_width: float = 2.0,
) -> np.ndarray:
    """Return ``[x, y, half_width, road_id]`` center samples."""
    if point_stride <= 0:
        raise ValueError("point_stride must be positive")
    if lane_half_width <= 0:
        raise ValueError("lane_half_width must be positive")
    road_map = _load_official_map(Path(path))
    rows = []
    for road_id, lanes in road_map.items():
        if not isinstance(lanes, Mapping):
            continue
        try:
            numeric_road_id = float(road_id)
        except (TypeError, ValueError):
            continue
        for lane_id, records in lanes.items():
            if isinstance(lane_id, str) or not isinstance(records, (tuple, list)):
                continue
            for record in records:
                if not isinstance(record, Mapping) or record.get("Type") != "Center":
                    continue
                points = record.get("Points", [])
                selected = list(points[::point_stride])
                if len(points) and (len(points) - 1) % point_stride != 0:
                    selected.append(points[-1])
                for point in selected:
                    location = _point_xyz(point)
                    rows.append(
                        (
                            float(location[0]),
                            -float(location[1]),
                            float(lane_half_width),
                            numeric_road_id,
                        )
                    )
    if not rows:
        raise ValueError(f"no Center lane points found in {path}")
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite lane point found in {path}")
    return result


def build_map_cache(
    map_files: Iterable[Path],
    output_path: Path,
    point_stride: int = 10,
    lane_half_width: float = 2.0,
) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    for map_path in map_files:
        match = _TOWN_RE.search(Path(map_path).name)
        if match is None:
            raise ValueError(f"cannot infer town name from {map_path}")
        town_name = match.group(1)
        cache[town_name] = compact_map_file(
            Path(map_path),
            point_stride=point_stride,
            lane_half_width=lane_half_width,
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as file_obj:
        pickle.dump(cache, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)
    return cache


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-stride", type=int, default=10)
    parser.add_argument("--lane-half-width", type=float, default=2.0)
    parser.add_argument(
        "--trusted-official-maps",
        action="store_true",
        help="required acknowledgement because official NPZ object arrays use pickle",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.trusted_official_maps:
        raise ValueError(
            "pass --trusted-official-maps only when --map-root contains "
            "the official rethinklab/Bench2Drive-Map files"
        )
    map_files = sorted(args.map_root.glob("Town*_HD_map.npz"))
    if not map_files:
        raise FileNotFoundError(f"no Town*_HD_map.npz files under {args.map_root}")
    cache = build_map_cache(
        map_files,
        args.output,
        point_stride=args.point_stride,
        lane_half_width=args.lane_half_width,
    )
    summary = ", ".join(f"{town}={len(points)}" for town, points in sorted(cache.items()))
    print(f"wrote {args.output}: {summary}")


if __name__ == "__main__":
    main()
