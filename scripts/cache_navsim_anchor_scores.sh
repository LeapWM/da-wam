#!/usr/bin/env bash
# Compute the complete NAVSIM v1 anchor labels on CPU; valid outputs are resumed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/environment/activate_model.sh"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$ROOT/exp}"
export DA_WAM_METRIC_CACHE_ROOT="${DA_WAM_METRIC_CACHE_ROOT:-$NAVSIM_EXP_ROOT/da_wam_cache}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$ROOT/navsim_v1"
exec python scripts/training/precompute_anchor_scores_v1.py \
  --max-workers "${ANCHOR_MAX_WORKERS:-8}" \
  --chunk-size "${ANCHOR_CHUNK_SIZE:-256}" \
  --verify-anchors 32 "$@"
