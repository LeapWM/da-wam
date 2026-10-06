#!/usr/bin/env bash
# E2-style short calibration from the newest Stage A best checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/c2-worldmodel/2639639/navsim_exp}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  SEARCH_ROOT="${NAVSIM_EXP_ROOT}/ke/PB_G_PostTrajFuture_StageA"
  latest_mtime=0
  latest_checkpoint=""
  while IFS= read -r candidate; do
    candidate_mtime="$(stat -c %Y "${candidate}")"
    if [[ "${candidate_mtime}" -gt "${latest_mtime}" ]]; then
      latest_mtime="${candidate_mtime}"
      latest_checkpoint="${candidate}"
    fi
  done < <(find "${SEARCH_ROOT}" -type f -name 'best_epoch=*.ckpt' -print 2>/dev/null || true)
  CHECKPOINT_PATH="${latest_checkpoint}"
fi

if [[ -z "${CHECKPOINT_PATH}" || ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "PostTraj Stage 2 requires a valid Stage A best checkpoint." >&2
  echo "Set CHECKPOINT_PATH explicitly after Stage A finishes." >&2
  exit 2
fi

SOURCE_CHECKPOINT_PATH="${CHECKPOINT_PATH}"
# Hydra treats an unquoted '=' inside an override value as grammar. Lightning
# checkpoint filenames use best_epoch=N.ckpt, so expose a grammar-safe path to
# the child runner while preserving the resolved source path in the log.
if [[ "${CHECKPOINT_PATH}" == *"="* ]]; then
  SAFE_CHECKPOINT_DIR="$(mktemp -d /tmp/posttraj_stage2_parent.XXXXXX)"
  SAFE_CHECKPOINT_PATH="${SAFE_CHECKPOINT_DIR}/parent.ckpt"
  ln -s "${CHECKPOINT_PATH}" "${SAFE_CHECKPOINT_PATH}"
  CHECKPOINT_PATH="${SAFE_CHECKPOINT_PATH}"
fi

export CHECKPOINT_PATH
export PROPOSAL_NUM="${PROPOSAL_NUM:-32}"
export POSTTRAJ_STAGE2_ENABLED=true
export POSTTRAJ_JOINT_ENABLED=false
export POSTTRAJ_FUTURE_LOSS_WEIGHT=0
export POSTTRAJ_STAGE2_NUM_LOCAL_HARD="${POSTTRAJ_STAGE2_NUM_LOCAL_HARD:-64}"
export POSTTRAJ_STAGE2_NUM_BALANCED="${POSTTRAJ_STAGE2_NUM_BALANCED:-64}"
export POSTTRAJ_STAGE2_USE_ANCHORS="${POSTTRAJ_STAGE2_USE_ANCHORS:-true}"
export POSTTRAJ_STAGE2_ANCHOR_LOSS_WEIGHT="${POSTTRAJ_STAGE2_ANCHOR_LOSS_WEIGHT:-0.5}"
export POSTTRAJ_FINAL_TOPK_RANK_WEIGHT="${POSTTRAJ_FINAL_TOPK_RANK_WEIGHT:-0}"
export MAX_EPOCHS="${MAX_EPOCHS:-3}"
export LR="${LR:-1e-5}"
export CHECKPOINT_SAVE_TOP_K="${CHECKPOINT_SAVE_TOP_K:-3}"
export CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-epoch}"
export CHECKPOINT_MODE="${CHECKPOINT_MODE:-max}"
export CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-PB_G_PostTrajFuture_Stage2_E2}"

echo "POSTTRAJ STAGE 2 parent: ${SOURCE_CHECKPOINT_PATH}"
echo "POSTTRAJ STAGE 2 Hydra-safe parent: ${CHECKPOINT_PATH}"
exec bash "${SCRIPT_DIR}/run_posttraj_future_stageA.sh"
