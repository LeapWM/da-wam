"""Validate the immutable cache bundle before starting GPU training."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, payload):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def validate(write_stamp=True):
    base = Path(os.environ["B2D_DENSE_INDEX"])
    bundle = Path(os.environ["B2D_CACHE_BUNDLE"])
    index_path = base / "index.json"
    manifest_path = bundle / "manifest.json"
    index = load(index_path)
    manifest = load(manifest_path)
    progress_path = Path(manifest["progress_file"])
    preflight_path = progress_path.parent / "PREFLIGHT.json"
    progress = load(progress_path)
    preflight = load(preflight_path)

    # Recompute the current content-addressed roots. Mutual agreement between
    # old manifest/progress/preflight files alone cannot detect source drift.
    import hydra
    from omegaconf import OmegaConf
    from expert_cache import _signature
    from tail_data import DenseDataset, expand_index
    root = Path(os.environ.get("B2D_NEW_CACHE", manifest["data_root"]))
    expanded = expand_index(index)
    dataset = DenseDataset(root, expanded, "train", load_route_targets=False)
    current_source = hashlib.sha256(repr(tuple(str(getattr(dataset, name))
        for name in ("cache", "safety_cache", "tail_cache"))).encode()).hexdigest()
    config = Path(__file__).resolve().parents[1] / "agent_config.yaml"
    cfg = hydra.utils.instantiate(OmegaConf.load(config).config)
    current_expert = _signature(type("Agent", (), {"_config": cfg})())

    require(len(index["train"]) == 950, "expected 950 training clips")
    require(len(index["val"]) == 50, "expected 50 validation clips")
    require(sum(c["raw"] for split in ("train", "val") for c in index[split]) == 247656,
            "expected 247656 observed frames")
    require(index["contract"]["input_sampling_hz"] == 10, "expected 10 Hz observations")
    require(manifest.get("schema_version") == 2, "expected scalar-speed bundle schema 2")
    require(manifest.get("ego_status", {}).get("dimension") == 12, "expected 12D ego status")
    require(manifest.get("ego_status", {}).get("velocity_mode") == "scalar_speed",
            "expected scalar speed without a dummy lateral channel")
    require(manifest.get("future_seconds") == 3.0 and manifest.get("future_dt") == 0.5,
            "expected 3 s future at 0.5 s intervals")
    route = manifest.get("navigation_target_xy", {})
    require(route.get("status") == "complete" and route.get("frames_done") == 247656,
            "route-target cache is incomplete")
    require(progress.get("status") == "complete", "full cache is not complete")
    require(progress.get("failed") == 0, "full cache contains failures")
    require(progress.get("done") == progress.get("total") == 247656,
            "full-cache frame count mismatch")
    require(progress.get("expert_done") == progress.get("expert_total") == 217656,
            "expert-cache frame count mismatch")
    require(preflight.get("status") == "passed", "cache preflight did not pass")
    require(progress.get("source_signature") == preflight.get("source_signature"),
            "full cache does not match the numeric/safety/tail contract")
    require(progress.get("source_signature") == current_source,
            "numeric/safety/tail source changed after cache generation")
    require(progress.get("signature") == preflight.get("signature"),
            "full cache does not match the preflight scoring contract")
    require(manifest.get("expert_signature") == progress.get("signature"),
            "bundle expert labels do not match the completed scoring contract")
    require(manifest.get("expert_signature") == current_expert,
            "scoring code/config/map changed after expert-cache generation")

    components = manifest.get("components", {})
    required = ("observations", "numeric", "safety", "tail_numeric",
                "expert_labels", "route_targets", "generation")
    for name in required:
        link = bundle / name
        require(name in components, f"manifest component is missing: {name}")
        require(link.is_symlink(), f"bundle component is not a symlink: {name}")
        require(link.exists(), f"bundle component target is unavailable: {name}")
        require(link.resolve() == Path(components[name]).resolve(),
                f"bundle component points to the wrong cache: {name}")

    evidence = {
        "schema_version": 1,
        "status": "ready",
        "checked_at": time.time(),
        "bundle": str(bundle),
        "counts": {"observations": 247656, "expert": 217656, "tail": 30000},
        "source_signature": progress["source_signature"],
        "expert_signature": progress["signature"],
        "sha256": {
            "index": digest(index_path),
            "manifest": digest(manifest_path),
            "progress": digest(progress_path),
            "preflight": digest(preflight_path),
        },
    }
    if write_stamp:
        stamp = bundle / "TRAINING_READY.json"
        current = None
        if stamp.is_file():
            try:
                current = load(stamp)
            except (OSError, json.JSONDecodeError):
                pass
        stable = ("schema_version", "status", "bundle", "counts",
                  "source_signature", "expert_signature", "sha256")
        if current is None or any(current.get(key) != evidence[key] for key in stable):
            atomic_json(stamp, evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true",
                        help="return a concise non-ready result without a traceback")
    args = parser.parse_args()
    try:
        validate(write_stamp=not args.probe)
    except Exception as exc:
        if not args.probe:
            raise
        print(f"Training readiness: NOT READY ({type(exc).__name__}: {exc})")
        return 1
    print("Training readiness: PASS")
    print("split=950/50 observations=247656 complete_rule=217656 tail=30000")
    print("ego_status=12D scalar_speed future=3s/6points observation=10Hz")
    if args.probe:
        print("Completed cache bundle can be reused; cache generation will be skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
