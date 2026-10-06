#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/environment/activate_model.sh"
export NAVSIM_V2_ROOT="$ROOT/navsim_v2" CONDA_ROOT="$MODEL_ENV"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT/checkpoints/navsim_final.ckpt}" MODE=momentum
config_only=false
for arg in "$@"; do [[ "$arg" == --cfg* ]] && config_only=true; done
if [[ "$config_only" == false ]]; then
 : "${OUTPUT_DIR:?Set OUTPUT_DIR to a new results directory}"
 [[ ! -e "$OUTPUT_DIR" ]] || { echo 'OUTPUT_DIR already exists' >&2; exit 2; }
fi
exec bash "$ROOT/navsim_v2/scripts/evaluation/eval_posttraj_temporal_v22.sh" \
 "agent.config.pretrain_pt_path=$ROOT/checkpoints/vjepa2_1_vitl.pt" \
 temporal_rerank.weight=0.125 "output_dir=${OUTPUT_DIR:-/tmp/da_wam_v2_config_only}" "$@"
