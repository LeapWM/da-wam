#!/usr/bin/env python3
"""Build the complete NAVSIM v1 PDM score cache for all 8192 anchors.

The cache mirrors the metric-cache directory layout.  A metric cache at

    <metric_root>/<log>/unknown/<token>/metric_cache.pkl

is written as

    <output_root>/<log>/unknown/<token>/metric_cache.pkl.npy

Each output is float16 with shape [8192, 6] and column order
[NOC, DAC, EP, TTC, comfort, final].  Writes are atomic and valid existing
files are skipped, so the command can be safely resumed.

This script is intentionally independent from the training loop.  A training
job may invoke it as a pre-stage, but anchor labels must be complete before
anchor auxiliary supervision is enabled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import platform
import socket
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SCORE_HEAD_ORDER = ["noc", "dac", "ep", "ttc", "comfort", "final"]
SCHEMA_VERSION = 2
OUTPUT_DTYPE = np.dtype(np.float16)
EXPECTED_ANCHOR_COUNT = 8192
EXPECTED_SCORE_SHAPE = (EXPECTED_ANCHOR_COUNT, len(SCORE_HEAD_ORDER))

_ANCHORS: Optional[np.ndarray] = None
_ANCHOR_PATH: Optional[Path] = None
_SIMULATOR: Optional[Any] = None
_SCORER: Optional[Any] = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def load_anchor_trajectories(anchor_path: Path) -> np.ndarray:
    global _ANCHORS, _ANCHOR_PATH
    anchor_path = anchor_path.absolute()
    if _ANCHORS is not None and _ANCHOR_PATH == anchor_path:
        return _ANCHORS

    raw = np.load(anchor_path, mmap_mode="r")
    if raw.shape != (EXPECTED_ANCHOR_COUNT, 40, 3):
        raise ValueError(
            f"Expected raw anchors [8192,40,3], got {raw.shape} from {anchor_path}"
        )
    anchors = np.asarray(raw[:, 4::5], dtype=np.float32)
    if anchors.shape != (EXPECTED_ANCHOR_COUNT, 8, 3):
        raise ValueError(f"Expected sampled anchors [8192,8,3], got {anchors.shape}")
    if not np.isfinite(anchors).all():
        raise ValueError(f"Anchor file contains non-finite values: {anchor_path}")

    _ANCHORS = anchors
    _ANCHOR_PATH = anchor_path
    return anchors


def metric_relative_path(metric_cache_path: Path, metric_cache_root: Path) -> Path:
    """Return a stable path even when metadata CSV contains a stale mount prefix."""
    try:
        return metric_cache_path.relative_to(metric_cache_root)
    except ValueError:
        parts = metric_cache_path.parts
        marker = metric_cache_root.name
        if marker not in parts:
            raise ValueError(
                f"Metric path {metric_cache_path} is not under {metric_cache_root} "
                f"and does not contain directory marker {marker!r}"
            )
        marker_index = len(parts) - 1 - list(reversed(parts)).index(marker)
        return Path(*parts[marker_index + 1 :])


def remap_metric_cache_path(metric_cache_path: Path, metric_cache_root: Path) -> Path:
    return metric_cache_root / metric_relative_path(metric_cache_path, metric_cache_root)


def output_path_for_metric(
    metric_cache_path: Path, metric_cache_root: Path, output_root: Path
) -> Path:
    relative = metric_relative_path(metric_cache_path, metric_cache_root)
    return output_root / relative.parent / f"{relative.name}.npy"


def read_metric_paths(csv_path: Path, metric_cache_root: Path) -> List[Path]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "file_name" not in reader.fieldnames:
            raise ValueError(f"CSV must contain a file_name column: {csv_path}")
        paths = [
            remap_metric_cache_path(Path(row["file_name"]), metric_cache_root)
            for row in reader
        ]
    if not paths:
        raise ValueError(f"No metric cache paths found in {csv_path}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"Duplicate metric-cache paths in {csv_path}")
    if len({path.parent.name for path in paths}) != len(paths):
        raise ValueError(f"Duplicate scene tokens in {csv_path}")
    return paths


def select_split(
    items: Sequence[Path], num_splits: int, split_index: int
) -> List[Path]:
    if num_splits <= 0:
        raise ValueError(f"num_splits must be positive, got {num_splits}")
    if not 0 <= split_index < num_splits:
        raise ValueError(
            f"split_index must be in [0, {num_splits - 1}], got {split_index}"
        )
    base, remainder = divmod(len(items), num_splits)
    start = split_index * base + min(split_index, remainder)
    size = base + int(split_index < remainder)
    return list(items[start : start + size])


def validate_cache_array(array: np.ndarray) -> Optional[str]:
    if array.shape != EXPECTED_SCORE_SHAPE:
        return f"shape={array.shape}, expected={EXPECTED_SCORE_SHAPE}"
    if array.dtype != OUTPUT_DTYPE:
        return f"dtype={array.dtype}, expected={OUTPUT_DTYPE}"
    if not np.isfinite(array).all():
        return "contains non-finite values"
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > 1.0:
        return f"range=[{minimum}, {maximum}], expected within [0,1]"
    return None


def valid_existing_cache(path: Path) -> Tuple[bool, Optional[str]]:
    if not path.exists():
        return False, "missing"
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        error = validate_cache_array(array)
        return error is None, error
    except Exception as exc:  # corrupted/partial files must never be silently used
        return False, f"{type(exc).__name__}: {exc}"


def scorer_runtime() -> Tuple[Any, Any]:
    global _SIMULATOR, _SCORER
    if _SIMULATOR is None or _SCORER is None:
        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )
        from navsim.agents.drive_jepa_perception_based.score_module.train_pdm_scorer import (
            PDMScorer,
            PDMScorerConfig,
        )
        from navsim.planning.simulation.planner.pdm_planner.simulation_v2.pdm_simulator import (
            PDMSimulator,
        )

        proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        _SIMULATOR = PDMSimulator(proposal_sampling)
        _SCORER = PDMScorer(proposal_sampling, PDMScorerConfig())
    return _SIMULATOR, _SCORER


def load_metric_cache(metric_cache_path: Path) -> Any:
    import lzma
    import pickle

    with lzma.open(metric_cache_path, "rb") as stream:
        return pickle.load(stream)


def simulate_poses(metric_cache: Any, poses: np.ndarray) -> np.ndarray:
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import fast_transform_trajectory

    simulator, _ = scorer_runtime()
    initial_ego_state = metric_cache.ego_state
    trajectory_states = [
        fast_transform_trajectory(
            Trajectory(model_trajectory),
            simulator.proposal_sampling,
            initial_ego_state,
        )
        for model_trajectory in poses
    ]
    trajectory_states_array = np.stack(trajectory_states, axis=0)
    extended = np.zeros(
        (*trajectory_states_array.shape[:-1], 11),
        dtype=trajectory_states_array.dtype,
    )
    extended[:, :, :3] = trajectory_states_array
    return simulator.simulate_proposals(extended, initial_ego_state)


def score_poses(metric_cache: Any, poses: np.ndarray) -> np.ndarray:
    """Score a pose batch with the exact v1 training PDM label implementation."""
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        MultiMetricIndex,
        WeightedMetricIndex,
    )

    _, scorer = scorer_runtime()
    simulated_states = simulate_poses(metric_cache, poses)
    final_scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.pdm_progress,
    )
    scores = np.stack(
        [
            scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, :],
            scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, :],
            scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, :],
            scorer._weighted_metrics[WeightedMetricIndex.TTC, :],
            scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, :],
            final_scores,
        ],
        axis=-1,
    )
    if scores.shape != (len(poses), len(SCORE_HEAD_ORDER)):
        raise RuntimeError(
            f"Unexpected v1 score shape {scores.shape} for {len(poses)} poses"
        )
    return np.asarray(scores, dtype=np.float64)


def score_all_anchors(
    metric_cache: Any, anchors: np.ndarray, chunk_size: int
) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    chunks = [
        score_poses(metric_cache, anchors[start : start + chunk_size])
        for start in range(0, len(anchors), chunk_size)
    ]
    scores = np.concatenate(chunks, axis=0)
    if scores.shape != EXPECTED_SCORE_SHAPE:
        raise RuntimeError(f"Unexpected complete score shape: {scores.shape}")
    if not np.isfinite(scores).all():
        raise RuntimeError("Online anchor scores contain non-finite values")
    if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
        raise RuntimeError(
            f"Online anchor score range [{scores.min()}, {scores.max()}] is outside [0,1]"
        )
    return scores


def stable_scene_seed(relative_path: Path, base_seed: int) -> int:
    digest = hashlib.sha256(str(relative_path).encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + base_seed) % (2**32)


def build_one_scene(task: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    metric_cache_path = Path(task["metric_cache_path"])
    metric_cache_root = Path(task["metric_cache_root"])
    output_root = Path(task["output_root"])
    output_path = output_path_for_metric(
        metric_cache_path, metric_cache_root, output_root
    )
    relative_path = metric_relative_path(metric_cache_path, metric_cache_root)

    valid, existing_error = valid_existing_cache(output_path)
    if valid and not task["overwrite"]:
        return {
            "status": "skipped",
            "relative_path": str(relative_path),
            "output_path": str(output_path),
            "elapsed_seconds": time.perf_counter() - started,
        }
    if output_path.exists() and not task["overwrite"]:
        raise RuntimeError(
            f"Invalid existing cache {output_path}: {existing_error}. "
            "Pass --overwrite to replace it."
        )
    if not metric_cache_path.is_file():
        raise FileNotFoundError(f"Metric cache not found: {metric_cache_path}")

    anchors = load_anchor_trajectories(Path(task["anchor_path"]))
    metric_cache = load_metric_cache(metric_cache_path)
    scores = score_all_anchors(metric_cache, anchors, int(task["chunk_size"]))

    chunk_compare_max_abs = None
    comparison_chunk_size = int(task["comparison_chunk_size"])
    if comparison_chunk_size:
        comparison_anchor_count = min(
            int(task["comparison_anchor_count"]), len(anchors)
        )
        comparison_anchors = anchors[:comparison_anchor_count]
        comparison_scores = np.concatenate(
            [
                score_poses(
                    metric_cache,
                    comparison_anchors[start : start + comparison_chunk_size],
                )
                for start in range(
                    0, comparison_anchor_count, comparison_chunk_size
                )
            ],
            axis=0,
        )
        primary_scores = scores[:comparison_anchor_count]
        chunk_compare_max_abs = float(
            np.max(np.abs(primary_scores - comparison_scores))
        )
        if not np.allclose(
            primary_scores, comparison_scores, rtol=0.0, atol=1e-10
        ):
            raise AssertionError(
                f"Chunk-size score mismatch for {relative_path}: "
                f"primary={task['chunk_size']} "
                f"comparison={comparison_chunk_size} "
                f"anchors={comparison_anchor_count} "
                f"max_abs={chunk_compare_max_abs}"
            )

    cached = scores.astype(OUTPUT_DTYPE)
    quantization_max_abs = float(
        np.max(np.abs(scores - cached.astype(np.float64)))
    )
    if quantization_max_abs > float(task["fp16_atol"]):
        raise AssertionError(
            f"fp16 quantization error {quantization_max_abs} exceeds "
            f"atol={task['fp16_atol']} for {relative_path}"
        )
    cache_error = validate_cache_array(cached)
    if cache_error is not None:
        raise AssertionError(f"Invalid cache array for {relative_path}: {cache_error}")

    verify_max_abs = None
    verify_count = min(int(task["verify_anchors"]), len(anchors))
    if verify_count:
        rng = np.random.default_rng(
            stable_scene_seed(relative_path, int(task["seed"]))
        )
        indices = np.sort(rng.choice(len(anchors), size=verify_count, replace=False))
        online = score_poses(metric_cache, anchors[indices])
        verify_max_abs = float(
            np.max(np.abs(online - cached[indices].astype(np.float64)))
        )
        if not np.allclose(
            online,
            cached[indices].astype(np.float64),
            rtol=0.0,
            atol=float(task["fp16_atol"]),
        ):
            raise AssertionError(
                f"Random online/cache mismatch for {relative_path}: "
                f"max_abs={verify_max_abs}"
            )

    atomic_save_npy(output_path, cached)
    return {
        "status": "written",
        "relative_path": str(relative_path),
        "output_path": str(output_path),
        "elapsed_seconds": time.perf_counter() - started,
        "quantization_max_abs": quantization_max_abs,
        "chunk_compare_max_abs": chunk_compare_max_abs,
        "verify_max_abs": verify_max_abs,
    }


def source_manifest(navsim_root: Path, anchor_path: Path) -> Dict[str, Any]:
    scorer_dir = (
        navsim_root
        / "navsim"
        / "agents"
        / "drive_jepa_perception_based"
        / "score_module"
    )
    source_paths = {
        "compute_navsim_score": scorer_dir / "compute_navsim_score.py",
        "train_pdm_scorer": scorer_dir / "train_pdm_scorer.py",
        "precompute_script": Path(__file__).absolute(),
    }
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
    from navsim.agents.drive_jepa_perception_based.score_module.train_pdm_scorer import (
        PDMScorerConfig,
    )

    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)

    raw = np.load(anchor_path, mmap_mode="r")
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "score_head_order": SCORE_HEAD_ORDER,
        "score_shape": list(EXPECTED_SCORE_SHAPE),
        "score_dtype": OUTPUT_DTYPE.name,
        "anchor": {
            "path": "data/8192.npy",
            "sha256": sha256_file(anchor_path),
            "raw_shape": list(raw.shape),
            "sampled_shape": [EXPECTED_ANCHOR_COUNT, 8, 3],
            "sampling": "raw[:, 4::5]",
        },
        "pdm_scorer_config": asdict(PDMScorerConfig()),
        "proposal_sampling": {
            "num_poses": proposal_sampling.num_poses,
            "interval_length": proposal_sampling.interval_length,
        },
        "source_sha256": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
    }
    signature_payload = json.dumps(
        cache_compatibility(payload), sort_keys=True
    ).encode("utf-8")
    payload["signature"] = hashlib.sha256(signature_payload).hexdigest()
    return payload


def cache_compatibility(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Compare label semantics, independently of machine paths and CLI edits.

    Accept the original schema-1 cache when its anchors, scoring sources and
    settings match. Keep the original manifest untouched when resuming.
    """
    return {
        key: manifest[key]
        for key in (
            "score_head_order", "score_shape", "score_dtype",
            "pdm_scorer_config", "proposal_sampling",
        )
    } | {
        "anchor": {
            key: manifest["anchor"][key]
            for key in ("sha256", "raw_shape", "sampled_shape", "sampling")
        },
        "scoring_sources": {
            key: manifest["source_sha256"][key]
            for key in ("compute_navsim_score", "train_pdm_scorer")
        },
    }


def ensure_manifest(
    output_root: Path, manifest: Dict[str, Any]
) -> Dict[str, Any]:
    path = output_root / "cache_manifest.json"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if cache_compatibility(existing) != cache_compatibility(manifest):
            raise RuntimeError(
                f"Cache manifest signature mismatch at {path}: "
                f"existing={existing.get('signature')} "
                f"current={manifest.get('signature')}"
            )
        return existing
    atomic_write_json(path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    navsim_root = Path(__file__).resolve().parents[2]
    exp_root = Path(
        os.environ.get("NAVSIM_EXP_ROOT", str(navsim_root.parent / "exp"))
    )
    cache_root = Path(
        os.environ.get("DA_WAM_METRIC_CACHE_ROOT", str(exp_root / "da_wam_cache"))
    )
    default_metric_root = cache_root / "train_metric_cache"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric-cache-root", type=Path, default=default_metric_root
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Defaults to METRIC_ROOT/metadata/train_metric_cache_metadata_node_0.csv.",
    )
    parser.add_argument(
        "--anchor-path", type=Path, default=navsim_root / "data" / "8192.npy"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=cache_root / "anchors_scores_v1_full_fp16",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Limit selected scenes for a pilot after applying the split.",
    )
    parser.add_argument(
        "--compare-scenes",
        type=int,
        default=0,
        help="Recompute a bounded anchor prefix with a second chunk size for the first N scenes.",
    )
    parser.add_argument(
        "--comparison-chunk-size",
        type=int,
        default=32,
        help="Safe alternate chunk size used by --compare-scenes.",
    )
    parser.add_argument(
        "--comparison-anchor-count",
        type=int,
        default=256,
        help="Bounded prefix size used for the alternate-chunk comparison.",
    )
    parser.add_argument(
        "--verify-anchors",
        type=int,
        default=0,
        help="Recompute this many deterministic random anchors per written scene.",
    )
    parser.add_argument("--fp16-atol", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print selected paths without scoring or writing outputs.",
    )
    args = parser.parse_args()
    if args.csv_path is None:
        args.csv_path = (
            args.metric_cache_root / "metadata"
            / "train_metric_cache_metadata_node_0.csv"
        )
    for key in ("metric_cache_root", "csv_path", "anchor_path", "output_root"):
        setattr(args, key, getattr(args, key).expanduser().absolute())
    return args


def main() -> int:
    args = parse_args()
    navsim_root = Path(__file__).resolve().parents[2]
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if args.compare_scenes < 0 or args.verify_anchors < 0:
        raise ValueError("validation counts must be non-negative")
    if not np.isfinite(args.fp16_atol) or not 0 < args.fp16_atol <= 5e-4:
        raise ValueError("--fp16-atol must be positive and at most 5e-4")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise ValueError("--max-scenes must be positive when provided")

    metric_paths = read_metric_paths(args.csv_path, args.metric_cache_root)
    metric_paths = select_split(metric_paths, args.num_splits, args.split_index)
    if args.max_scenes is not None:
        metric_paths = metric_paths[: args.max_scenes]
    if not metric_paths:
        raise ValueError("Selected split contains no scenes")
    if args.compare_scenes > len(metric_paths):
        raise ValueError(
            "--compare-scenes cannot exceed the number of selected scenes"
        )
    if args.compare_scenes and args.comparison_chunk_size <= 0:
        raise ValueError("--comparison-chunk-size must be positive")
    if args.compare_scenes and args.comparison_anchor_count <= 0:
        raise ValueError("--comparison-anchor-count must be positive")

    print(
        f"Selected {len(metric_paths)} scenes; split={args.split_index}/"
        f"{args.num_splits}; chunk_size={args.chunk_size}; workers={args.max_workers}"
    )
    print(f"Metric root: {args.metric_cache_root}")
    print(f"Output root: {args.output_root}")
    print(f"Anchor path: {args.anchor_path}")
    for path in metric_paths[:3]:
        print(f"  {path}")
    load_anchor_trajectories(args.anchor_path)
    missing_examples = [path for path in metric_paths[:3] if not path.is_file()]
    if missing_examples:
        raise FileNotFoundError(
            "Selected metric-cache paths do not exist after remapping: "
            + ", ".join(str(path) for path in missing_examples)
        )
    if args.dry_run:
        return 0
    manifest = source_manifest(navsim_root, args.anchor_path)
    manifest = ensure_manifest(args.output_root, manifest)

    tasks: List[Dict[str, Any]] = []
    for index, metric_path in enumerate(metric_paths):
        tasks.append(
            {
                "metric_cache_path": str(metric_path),
                "metric_cache_root": str(args.metric_cache_root),
                "output_root": str(args.output_root),
                "anchor_path": str(args.anchor_path),
                "chunk_size": args.chunk_size,
                "comparison_chunk_size": (
                    args.comparison_chunk_size if index < args.compare_scenes else 0
                ),
                "comparison_anchor_count": args.comparison_anchor_count,
                "verify_anchors": args.verify_anchors,
                "fp16_atol": args.fp16_atol,
                "seed": args.seed,
                "overwrite": args.overwrite,
            }
        )

    started_at = utc_now()
    wall_start = time.perf_counter()
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def record_result(task: Dict[str, Any], future: Any) -> None:
        try:
            result = future.result() if hasattr(future, "result") else future
            results.append(result)
            print(
                f"[{len(results) + len(errors)}/{len(tasks)}] "
                f"{result['status']} {result['relative_path']} "
                f"({result['elapsed_seconds']:.2f}s)",
                flush=True,
            )
        except Exception as exc:
            error = {
                "metric_cache_path": task["metric_cache_path"],
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            errors.append(error)
            print(
                f"[{len(results) + len(errors)}/{len(tasks)}] ERROR "
                f"{task['metric_cache_path']}: {error['error']}",
                file=sys.stderr,
                flush=True,
            )
            if args.fail_fast:
                raise

    if args.max_workers == 1:
        for task in tasks:
            future = concurrent.futures.Future()
            try:
                future.set_result(build_one_scene(task))
            except Exception as exc:
                future.set_exception(exc)
            try:
                record_result(task, future)
            except Exception:
                break
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            # Keep only a small bounded queue of futures.  The full training
            # metadata has O(1e5) scenes, and eagerly submitting all of them
            # wastes substantial memory before any score work starts.
            task_iterator = iter(tasks)
            future_to_task: Dict[Any, Dict[str, Any]] = {}
            max_in_flight = min(len(tasks), args.max_workers * 2)
            for _ in range(max_in_flight):
                task = next(task_iterator)
                future_to_task[executor.submit(build_one_scene, task)] = task

            abort = False
            while future_to_task and not abort:
                done, _ = concurrent.futures.wait(
                    tuple(future_to_task),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    task = future_to_task.pop(future)
                    try:
                        record_result(task, future)
                    except Exception:
                        for pending in future_to_task:
                            pending.cancel()
                        abort = True
                        break

                    try:
                        next_task = next(task_iterator)
                    except StopIteration:
                        continue
                    future_to_task[
                        executor.submit(build_one_scene, next_task)
                    ] = next_task

    elapsed = time.perf_counter() - wall_start
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "cache_signature": manifest["signature"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": elapsed,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "selected_scenes": len(tasks),
        "written": sum(result["status"] == "written" for result in results),
        "skipped": sum(result["status"] == "skipped" for result in results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
    }
    run_name = (
        f"run_split{args.split_index:04d}-of-{args.num_splits:04d}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
    )
    atomic_write_json(args.output_root / "runs" / run_name, run_manifest)
    print(
        f"Finished: written={run_manifest['written']} "
        f"skipped={run_manifest['skipped']} failed={run_manifest['failed']} "
        f"elapsed={elapsed:.1f}s"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
