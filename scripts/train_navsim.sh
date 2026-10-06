#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/environment/activate_model.sh"
cd "$ROOT/navsim_v1"
case "${1:-}" in
 stage1) shift; exec bash scripts/training/run_posttraj_future_joint30.sh "$@" ;;
 stage2) shift; : "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to your Stage 1 epoch 15 checkpoint}"; export CHECKPOINT_PATH; exec bash scripts/training/run_posttraj_joint_e15_safety_final.sh "$@" ;;
 *) echo 'Usage: train_navsim.sh stage1|stage2' >&2;exit 2 ;;
esac
