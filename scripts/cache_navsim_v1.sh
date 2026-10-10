#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/c2-worldmodel/2639639/navsim_exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/c2-worldmodel/training_data/OpenScene/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/train_da_wam_future_cache}"
export CACHE_WORKERS="${CACHE_WORKERS:-16}"
export FORCE_CACHE_COMPUTATION="${FORCE_CACHE_COMPUTATION:-false}"
# Supported training horizons: [1], [1,2], [1,2,3], [1,2,3,4].
# Use a separate CACHE_PATH for each cached offset list (or reuse a superset
# cache during training). Changing offsets does not invalidate existing files.
export POSTTRAJ_FUTURE_FRAME_OFFSETS="${POSTTRAJ_FUTURE_FRAME_OFFSETS:-[1]}"

[[ -d "${OPENSCENE_DATA_ROOT}/navsim_logs/trainval" ]] || {
  echo "FATAL: NAVSIM logs not found: ${OPENSCENE_DATA_ROOT}/navsim_logs/trainval" >&2
  exit 1
}
[[ -d "${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval" ]] || {
  echo "FATAL: OpenScene sensor blobs not found: ${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval" >&2
  exit 1
}
[[ "${CACHE_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: CACHE_WORKERS must be a positive integer: ${CACHE_WORKERS}" >&2
  exit 1
}
[[ "${FORCE_CACHE_COMPUTATION}" == "true" || "${FORCE_CACHE_COMPUTATION}" == "false" ]] || {
  echo "FATAL: FORCE_CACHE_COMPUTATION must be true or false" >&2
  exit 1
}

source "${ROOT}/environment/activate_model.sh"

# cache_data=true bypasses model construction, so validate its horizon contract
# here before expensive caching rather than waiting for training to reject it.
POSTTRAJ_FUTURE_FRAME_OFFSETS="$(python - "${POSTTRAJ_FUTURE_FRAME_OFFSETS}" <<'PY_OFFSETS'
import json
import sys

message = "FATAL: POSTTRAJ_FUTURE_FRAME_OFFSETS must be [1], [1,2], [1,2,3], or [1,2,3,4]"
try:
    offsets = json.loads(sys.argv[1])
except (ValueError, TypeError):
    raise SystemExit(message)
if (not isinstance(offsets, list)
        or not 1 <= len(offsets) <= 4
        or any(type(value) is not int for value in offsets)
        or offsets != list(range(1, len(offsets) + 1))):
    raise SystemExit(message)
print(json.dumps(offsets, separators=(",", ":")))
PY_OFFSETS
)"
case "${POSTTRAJ_FUTURE_FRAME_OFFSETS}" in
  '[1]') FUTURE_HORIZONS=1 ;;
  '[1,2]') FUTURE_HORIZONS=2 ;;
  '[1,2,3]') FUTURE_HORIZONS=3 ;;
  '[1,2,3,4]') FUTURE_HORIZONS=4 ;;
esac
cd "${NAVSIM_DEVKIT_ROOT}"

CMD=(
  python navsim/planning/script/run_dataset_caching.py
  agent=drive_jepa_perception_based_agent
  agent.cache_data=true
  agent.config.posttraj_future_enabled=true
  "agent.config.posttraj_future_frame_offsets=${POSTTRAJ_FUTURE_FRAME_OFFSETS}"
  train_test_split=navtrain
  split=trainval
  worker=single_machine_thread_pool
  "worker.max_workers=${CACHE_WORKERS}"
  gpu=false
  "cache_path=${CACHE_PATH}"
  use_cache_without_dataset=false
  "force_cache_computation=${FORCE_CACHE_COMPUTATION}"
  experiment_name=da_wam_future_target_cache
  "$@"
)

echo "NAVSIM train/val cache: ${CACHE_PATH}"
echo "future camera offsets: ${POSTTRAJ_FUTURE_FRAME_OFFSETS} (${FUTURE_HORIZONS} frames, nominal 0.5 s per offset)"
echo "worker threads: ${CACHE_WORKERS}"
echo "force recompute: ${FORCE_CACHE_COMPUTATION}"
printf '+'
printf ' %q' "${CMD[@]}"
echo
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  CONFIG_DUMP="$(mktemp)"
  if ! "${CMD[@]}" --cfg job >"${CONFIG_DUMP}"; then
    rm -f "${CONFIG_DUMP}"
    exit 1
  fi
  grep -E '^[[:space:]]+(posttraj_future_enabled|posttraj_future_frame_offsets|cache_data):|^cache_path:|^use_cache_without_dataset:|^force_cache_computation:' "${CONFIG_DUMP}"
  rm -f "${CONFIG_DUMP}"
  echo "DRY-RUN: Hydra config validated; cache computation was not started."
  exit 0
fi

mkdir -p "${CACHE_PATH}"
"${CMD[@]}"

python - "${CACHE_PATH}" "${POSTTRAJ_FUTURE_FRAME_OFFSETS}" <<'PY'
import ast
import gzip
import pickle
import sys
from pathlib import Path

import torch

cache_root = Path(sys.argv[1])
offsets = tuple(int(value) for value in ast.literal_eval(sys.argv[2]))
sample = next(cache_root.glob("*/*/da_wam_target.gz"), None)
if sample is None:
    raise SystemExit(f"FATAL: no da_wam_target.gz files found under {cache_root}")

with gzip.open(sample, "rb") as handle:
    targets = pickle.load(handle)

future = targets.get("future_camera_features")
order = targets.get("future_camera_offset_order")
if not isinstance(future, torch.Tensor):
    raise SystemExit(f"FATAL: {sample} has no tensor future_camera_features")
if tuple(future.shape) != (len(offsets), 3, 256, 512):
    raise SystemExit(
        f"FATAL: expected future camera shape {(len(offsets), 3, 256, 512)}, got {tuple(future.shape)}"
    )
if not isinstance(order, torch.Tensor) or tuple(int(v) for v in order.tolist()) != offsets:
    raise SystemExit(f"FATAL: cached future offsets do not match requested offsets {offsets}")

print(f"Verified {sample.relative_to(cache_root)}: future_camera_features={tuple(future.shape)}, offsets={offsets}")
PY
