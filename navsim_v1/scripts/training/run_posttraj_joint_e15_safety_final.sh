#!/usr/bin/env bash
# Final-head-only safety-focused scorer continuation from Joint30 e15.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/c2-worldmodel/2639639/navsim_exp/ke/PB_G_PostTrajFuture_Joint30/08.16_00.06/checkpoints/best_epoch=15.ckpt}"
export PROPOSAL_NUM=32
export POSTTRAJ_STAGE2_USE_ANCHORS=true
export POSTTRAJ_STAGE2_NUM_LOCAL_HARD=32
export POSTTRAJ_STAGE2_NUM_BALANCED=96
export POSTTRAJ_STAGE2_SAMPLING_PROFILE=safety_focus
export POSTTRAJ_STAGE2_ANCHOR_LOSS_WEIGHT=0.5

# Keep the deployed six-logit architecture, but supervise only final logit 5.
export POSTTRAJ_FINAL_TOPK_RANK_WEIGHT=0
export POSTTRAJ_SAFETY_HARD_ENABLED=true
export POSTTRAJ_SAFETY_UNSAFE_FINAL_WEIGHT="${POSTTRAJ_SAFETY_UNSAFE_FINAL_WEIGHT:-4.0}"
export POSTTRAJ_SAFETY_RANK_WEIGHT="${POSTTRAJ_SAFETY_RANK_WEIGHT:-0.1}"
export POSTTRAJ_SAFETY_THRESHOLD="${POSTTRAJ_SAFETY_THRESHOLD:-0.95}"
export POSTTRAJ_SAFETY_RANK_TOPK="${POSTTRAJ_SAFETY_RANK_TOPK:-8}"
export POSTTRAJ_SAFETY_RANK_MARGIN="${POSTTRAJ_SAFETY_RANK_MARGIN:-0.05}"

export MAX_EPOCHS="${MAX_EPOCHS:-3}"
export LR="${LR:-1e-6}"
export CHECKPOINT_SAVE_TOP_K="${CHECKPOINT_SAVE_TOP_K:-3}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-PB_G_PostTraj_JointE15_SafetyFinal_CF128_3ep}"

exec bash "${SCRIPT_DIR}/run_posttraj_future_stage2.sh"
