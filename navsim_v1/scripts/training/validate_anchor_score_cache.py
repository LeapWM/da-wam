#!/usr/bin/env python3
"""Validate full NAVSIM v1 anchor-cache coverage, format and optional SHA256.

Use the published files.jsonl index or supply --metric-csv for a locally
computed cache. All expected files are checked for existence and size.
By default all arrays are checked; --sample-files N limits content checks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
from pathlib import Path

import numpy as np

from precompute_anchor_scores_v1 import (
    EXPECTED_SCORE_SHAPE,
    OUTPUT_DTYPE,
    SCORE_HEAD_ORDER,
    metric_relative_path,
    read_metric_paths,
    sha256_file,
    validate_cache_array,
)


def relative_file(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a relative cache path: {value}")
    if path.name != "metric_cache.pkl.npy":
        raise ValueError(f"Unexpected score filename: {value}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--metric-csv", type=Path)
    parser.add_argument("--metric-cache-root", type=Path)
    parser.add_argument(
        "--anchor-path", type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "8192.npy",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--sample-files", type=int, default=0,
        help="Check N arrays instead of all arrays; 0 checks all.",
    )
    parser.add_argument(
        "--verify-sha256", action="store_true",
        help="Also verify per-file hashes from the published files.jsonl.",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.sample_files < 0:
        parser.error("--workers must be positive; --sample-files cannot be negative")
    root = args.cache_root.expanduser().absolute()
    manifest = json.loads((root / "cache_manifest.json").read_text())
    if (
        manifest["score_shape"] != list(EXPECTED_SCORE_SHAPE)
        or manifest["score_dtype"] != OUTPUT_DTYPE.name
        or manifest["score_head_order"] != SCORE_HEAD_ORDER
    ):
        raise ValueError("Unexpected score schema in cache_manifest.json")
    if sha256_file(args.anchor_path) != manifest["anchor"]["sha256"]:
        raise ValueError("Anchor SHA256 does not match the score cache")

    entries = {}
    index_path = root / "files.jsonl"
    if index_path.is_file():
        expected_index_hash = manifest.get("file_index_sha256")
        if expected_index_hash and sha256_file(index_path) != expected_index_hash:
            raise ValueError("files.jsonl SHA256 does not match the manifest")
        for line in index_path.read_text().splitlines():
            item = json.loads(line)
            path = relative_file(item["path"])
            if path in entries:
                raise ValueError(f"Duplicate file in index: {path}")
            if item["token"] != path.parent.name:
                raise ValueError(f"Token/path mismatch: {path}")
            entries[path] = item
        if manifest.get("scene_count") != len(entries):
            raise ValueError("Index count differs from manifest.scene_count")
    if args.metric_csv:
        metric_root = args.metric_cache_root or args.metric_csv.parent.parent
        paths = read_metric_paths(args.metric_csv, metric_root)
        expected = {
            relative_file(str(metric_relative_path(path, metric_root)) + ".npy")
            for path in paths
        }
        if entries:
            missing = expected - entries.keys()
            if missing:
                raise ValueError(f"{len(missing)} required metric-cache scenes absent from index")
        else:
            entries = {path: {"path": str(path)} for path in expected}
    if not entries:
        raise ValueError("Provide files.jsonl or --metric-csv with expected scene paths")
    if len({path.parent.name for path in entries}) != len(entries):
        raise ValueError("Duplicate scene tokens in expected cache")
    if args.verify_sha256 and any("sha256" not in item for item in entries.values()):
        raise ValueError("--verify-sha256 requires the published per-file index")

    ordered = sorted(entries)
    selected = set(
        random.Random(20261020).sample(ordered, min(args.sample_files, len(ordered)))
        if args.sample_files else ordered
    )

    def check(path: Path) -> None:
        full_path = root / path
        item = entries[path]
        size = full_path.stat().st_size
        # The released float16 arrays have a 128-byte NPY header.
        expected_size = item.get("bytes", 128 + int(np.prod(EXPECTED_SCORE_SHAPE)) * 2)
        if size != expected_size:
            raise ValueError(f"Wrong file size: {path}: {size} != {expected_size}")
        if path in selected:
            array = np.load(full_path, allow_pickle=False)
            error = validate_cache_array(array)
            if error:
                raise ValueError(f"{path}: {error}")
        if args.verify_sha256 and sha256_file(full_path) != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, _ in enumerate(executor.map(check, ordered), 1):
            if index % 10000 == 0:
                print(f"Checked {index}/{len(ordered)} files", flush=True)
    print(
        f"VALID: {len(ordered)} unique scenes; all files present with expected sizes; "
        f"{len(selected)} arrays checked; SHA256={'all' if args.verify_sha256 else 'not requested'}"
    )


if __name__ == "__main__":
    main()
