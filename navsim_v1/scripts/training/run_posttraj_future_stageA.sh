#!/usr/bin/env bash
# PB-G matched Stage A with post-trajectory candidate future prediction.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/c2-worldmodel/2639639/navsim_exp}"
OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/c2-worldmodel/training_data/OpenScene/dataset}"
CONDA_SH="${CONDA_SH:-/home/myuser/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-drive-jepa}"

NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PROPOSAL_NUM="${PROPOSAL_NUM:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-32}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
POSTTRAJ_STAGE2_ENABLED="${POSTTRAJ_STAGE2_ENABLED:-false}"
POSTTRAJ_FUTURE_LOSS_WEIGHT="${POSTTRAJ_FUTURE_LOSS_WEIGHT:-0.1}"
POSTTRAJ_FUTURE_FRAME_OFFSETS="${POSTTRAJ_FUTURE_FRAME_OFFSETS:-[1]}"
POSTTRAJ_FULL_CANDIDATE_LATENT="${POSTTRAJ_FULL_CANDIDATE_LATENT:-false}"
POSTTRAJ_STAGE2_ANCHOR_CACHE_ROOT="${POSTTRAJ_STAGE2_ANCHOR_CACHE_ROOT:-}"
POSTTRAJ_STAGE2_NUM_LOCAL_HARD="${POSTTRAJ_STAGE2_NUM_LOCAL_HARD:-64}"
POSTTRAJ_STAGE2_NUM_BALANCED="${POSTTRAJ_STAGE2_NUM_BALANCED:-64}"
POSTTRAJ_STAGE2_ANCHOR_LOSS_WEIGHT="${POSTTRAJ_STAGE2_ANCHOR_LOSS_WEIGHT:-0.5}"
POSTTRAJ_STAGE2_USE_ANCHORS="${POSTTRAJ_STAGE2_USE_ANCHORS:-true}"
POSTTRAJ_STAGE2_SAMPLING_PROFILE="${POSTTRAJ_STAGE2_SAMPLING_PROFILE:-balanced}"
POSTTRAJ_FINAL_TOPK_RANK_WEIGHT="${POSTTRAJ_FINAL_TOPK_RANK_WEIGHT:-0}"
POSTTRAJ_FINAL_TOPK_RANK_K="${POSTTRAJ_FINAL_TOPK_RANK_K:-8}"
POSTTRAJ_FINAL_TOPK_RANK_SCORE_GAP="${POSTTRAJ_FINAL_TOPK_RANK_SCORE_GAP:-0.01}"
POSTTRAJ_FINAL_TOPK_RANK_MARGIN_CAP="${POSTTRAJ_FINAL_TOPK_RANK_MARGIN_CAP:-0.05}"
POSTTRAJ_FINAL_TOPK_RANK_FALSE_TOPK_WEIGHT="${POSTTRAJ_FINAL_TOPK_RANK_FALSE_TOPK_WEIGHT:-2.0}"
POSTTRAJ_FINAL_TOPK_RANK_SAFETY_WEIGHT="${POSTTRAJ_FINAL_TOPK_RANK_SAFETY_WEIGHT:-3.0}"
POSTTRAJ_FINAL_TOPK_RANK_SAFETY_THRESHOLD="${POSTTRAJ_FINAL_TOPK_RANK_SAFETY_THRESHOLD:-0.95}"
POSTTRAJ_SAFETY_HARD_ENABLED="${POSTTRAJ_SAFETY_HARD_ENABLED:-false}"
POSTTRAJ_SAFETY_RANK_WEIGHT="${POSTTRAJ_SAFETY_RANK_WEIGHT:-0}"
POSTTRAJ_SAFETY_THRESHOLD="${POSTTRAJ_SAFETY_THRESHOLD:-0.95}"
POSTTRAJ_SAFETY_RANK_TOPK="${POSTTRAJ_SAFETY_RANK_TOPK:-8}"
POSTTRAJ_SAFETY_RANK_MARGIN="${POSTTRAJ_SAFETY_RANK_MARGIN:-0.05}"
POSTTRAJ_SAFETY_UNSAFE_FINAL_WEIGHT="${POSTTRAJ_SAFETY_UNSAFE_FINAL_WEIGHT:-4.0}"
POSTTRAJ_SAFE_FINAL_RANK_WEIGHT="${POSTTRAJ_SAFE_FINAL_RANK_WEIGHT:-0}"
POSTTRAJ_SAFE_FINAL_RANK_TOPK="${POSTTRAJ_SAFE_FINAL_RANK_TOPK:-8}"
POSTTRAJ_SAFE_FINAL_RANK_SCORE_GAP="${POSTTRAJ_SAFE_FINAL_RANK_SCORE_GAP:-0.01}"
POSTTRAJ_SAFE_FINAL_RANK_MARGIN_CAP="${POSTTRAJ_SAFE_FINAL_RANK_MARGIN_CAP:-0.05}"
POSTTRAJ_STAGE2_SCHEDULE_ENABLED="${POSTTRAJ_STAGE2_SCHEDULE_ENABLED:-false}"
POSTTRAJ_STAGE2_UNSAFE_FINAL_WEIGHT_SCHEDULE="${POSTTRAJ_STAGE2_UNSAFE_FINAL_WEIGHT_SCHEDULE:-[]}"
POSTTRAJ_STAGE2_SAFETY_RANK_WEIGHT_SCHEDULE="${POSTTRAJ_STAGE2_SAFETY_RANK_WEIGHT_SCHEDULE:-[]}"
POSTTRAJ_STAGE2_SAFE_FINAL_RANK_WEIGHT_SCHEDULE="${POSTTRAJ_STAGE2_SAFE_FINAL_RANK_WEIGHT_SCHEDULE:-[]}"
POSTTRAJ_STAGE2_LR_SCHEDULE="${POSTTRAJ_STAGE2_LR_SCHEDULE:-[]}"
POSTTRAJ_JOINT_ENABLED="${POSTTRAJ_JOINT_ENABLED:-false}"
POSTTRAJ_JOINT_WARMUP_EPOCHS="${POSTTRAJ_JOINT_WARMUP_EPOCHS:-5}"
POSTTRAJ_JOINT_RAMP_EPOCHS="${POSTTRAJ_JOINT_RAMP_EPOCHS:-5}"
POSTTRAJ_JOINT_ANCHOR_LOSS_WEIGHT="${POSTTRAJ_JOINT_ANCHOR_LOSS_WEIGHT:-0.25}"
POSTTRAJ_JOINT_SAFETY_SCHEDULE_ENABLED="${POSTTRAJ_JOINT_SAFETY_SCHEDULE_ENABLED:-false}"
POSTTRAJ_JOINT_UNSAFE_FINAL_WEIGHT_SCHEDULE="${POSTTRAJ_JOINT_UNSAFE_FINAL_WEIGHT_SCHEDULE:-[]}"
POSTTRAJ_JOINT_SAFETY_RANK_WEIGHT_SCHEDULE="${POSTTRAJ_JOINT_SAFETY_RANK_WEIGHT_SCHEDULE:-[]}"
POSTTRAJ_JOINT_SAFE_FINAL_RANK_WEIGHT_SCHEDULE="${POSTTRAJ_JOINT_SAFE_FINAL_RANK_WEIGHT_SCHEDULE:-[]}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp_find_unused_parameters_true}"
TRAINER_PRECISION="${TRAINER_PRECISION:-bf16}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
CHECKPOINT_SAVE_TOP_K="${CHECKPOINT_SAVE_TOP_K:-}"
CHECKPOINT_SAVE_LAST="${CHECKPOINT_SAVE_LAST:-}"
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-}"
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-}"
LOG_NAME_OVERRIDE="${LOG_NAME_OVERRIDE:-}"
CACHE_PATH="${CACHE_PATH:-${NAVSIM_EXP_ROOT}/train_da_wam_future_cache}"
PRETRAIN_PT_PATH="${PRETRAIN_PT_PATH:-/mnt/downloads-1/models/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt}"
ANCHOR_TRAJECTORY_PATH="${ANCHOR_TRAJECTORY_PATH:-${NAVSIM_DEVKIT_ROOT}/../navsim_v2/data/8192.npy}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-PB_G_PostTrajFuture_StageA}"

[[ -f "${CONDA_SH}" ]] || {
  echo "FATAL: conda init not found: ${CONDA_SH}" >&2
  exit 1
}
[[ -f "${PRETRAIN_PT_PATH}" ]] || {
  echo "FATAL: V-JEPA checkpoint not found: ${PRETRAIN_PT_PATH}" >&2
  exit 1
}
[[ -f "${ANCHOR_TRAJECTORY_PATH}" ]] || {
  echo "FATAL: trajectory anchor bank not found: ${ANCHOR_TRAJECTORY_PATH}" >&2
  exit 1
}
[[ -d "${CACHE_PATH}" ]] || {
  echo "FATAL: PB cache not found: ${CACHE_PATH}" >&2
  exit 1
}
[[ "${PROPOSAL_NUM}" =~ ^[1-9][0-9]*$ ]] || {
  echo "FATAL: PROPOSAL_NUM must be a positive integer: ${PROPOSAL_NUM}" >&2
  exit 1
}

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# The host image also provides PyTorch native libraries.  Keep the activated
# conda build first to avoid mixing its CUDA runtime with the host build.
CONDA_TORCH_LIB="${CONDA_PREFIX}/lib/python3.9/site-packages/torch/lib"
export LD_LIBRARY_PATH="${CONDA_TORCH_LIB}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export NAVSIM_DEVKIT_ROOT NAVSIM_EXP_ROOT OPENSCENE_DATA_ROOT
export NUPLAN_DATA_ROOT="${OPENSCENE_DATA_ROOT}"
export NUPLAN_MAPS_ROOT="${OPENSCENE_DATA_ROOT}/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${NAVSIM_DEVKIT_ROOT}/vjepa2:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

cd "${NAVSIM_DEVKIT_ROOT}"

STAGE_LABEL="STAGE A"
if [[ "${POSTTRAJ_STAGE2_ENABLED}" == "true" ]]; then
  STAGE_LABEL="STAGE 2 · E2 CALIBRATION"
elif [[ "${POSTTRAJ_JOINT_ENABLED}" == "true" ]]; then
  STAGE_LABEL="STAGE A + E2 JOINT"
fi
echo "============================================================"
echo "PB-G POST-TRAJECTORY FUTURE · ${STAGE_LABEL}"
echo "============================================================"
echo "root             : ${NAVSIM_DEVKIT_ROOT}"
echo "experiment       : ${EXPERIMENT_NAME}"
echo "GPUs / batch     : ${NUM_GPUS} / ${BATCH_SIZE} per GPU"
echo "proposals        : ${PROPOSAL_NUM}"
echo "epochs / LR      : ${MAX_EPOCHS} / ${LR}"
echo "LoRA rank        : ${LORA_RANK}"
echo "checkpoint       : ${CHECKPOINT_PATH:-from scratch}"
echo "stage2           : ${POSTTRAJ_STAGE2_ENABLED}"
echo "joint            : ${POSTTRAJ_JOINT_ENABLED}"
echo "future target    : adjacent EMA pairs, offsets=${POSTTRAJ_FUTURE_FRAME_OFFSETS}, mean loss weight=${POSTTRAJ_FUTURE_LOSS_WEIGHT}"
echo "future repr      : $([[ "${POSTTRAJ_FULL_CANDIDATE_LATENT}" == "true" ]] && echo 'full [P,256,16,32]' || echo 'compact [P,8,256]')"
echo "selection        : learned final head"
if [[ "${POSTTRAJ_STAGE2_ENABLED}" == "true" ]]; then
  echo "anchors          : ${POSTTRAJ_STAGE2_USE_ANCHORS}, weight=${POSTTRAJ_STAGE2_ANCHOR_LOSS_WEIGHT}"
  echo "anchor profile   : ${POSTTRAJ_STAGE2_SAMPLING_PROFILE} (${POSTTRAJ_STAGE2_NUM_LOCAL_HARD}+${POSTTRAJ_STAGE2_NUM_BALANCED})"
  echo "final top-k rank : k=${POSTTRAJ_FINAL_TOPK_RANK_K}, weight=${POSTTRAJ_FINAL_TOPK_RANK_WEIGHT}"
  echo "safety final     : enabled=${POSTTRAJ_SAFETY_HARD_ENABLED}, unsafe_weight=${POSTTRAJ_SAFETY_UNSAFE_FINAL_WEIGHT}, rank_weight=${POSTTRAJ_SAFETY_RANK_WEIGHT}"
  echo "safe final rank  : k=${POSTTRAJ_SAFE_FINAL_RANK_TOPK}, weight=${POSTTRAJ_SAFE_FINAL_RANK_WEIGHT}"
  echo "stage2 schedule  : enabled=${POSTTRAJ_STAGE2_SCHEDULE_ENABLED}"
  if [[ "${POSTTRAJ_STAGE2_SCHEDULE_ENABLED}" == "true" ]]; then
    echo "unsafe schedule  : ${POSTTRAJ_STAGE2_UNSAFE_FINAL_WEIGHT_SCHEDULE}"
    echo "safety schedule  : ${POSTTRAJ_STAGE2_SAFETY_RANK_WEIGHT_SCHEDULE}"
    echo "safe-rank sched  : ${POSTTRAJ_STAGE2_SAFE_FINAL_RANK_WEIGHT_SCHEDULE}"
    echo "LR schedule      : ${POSTTRAJ_STAGE2_LR_SCHEDULE}"
  fi
elif [[ "${POSTTRAJ_JOINT_ENABLED}" == "true" ]]; then
  echo "counterfactual   : warmup ${POSTTRAJ_JOINT_WARMUP_EPOCHS}, ramp ${POSTTRAJ_JOINT_RAMP_EPOCHS}, target ${POSTTRAJ_JOINT_ANCHOR_LOSS_WEIGHT}"
  echo "safety schedule  : enabled=${POSTTRAJ_JOINT_SAFETY_SCHEDULE_ENABLED}"
  if [[ "${POSTTRAJ_JOINT_SAFETY_SCHEDULE_ENABLED}" == "true" ]]; then
    echo "unsafe schedule  : ${POSTTRAJ_JOINT_UNSAFE_FINAL_WEIGHT_SCHEDULE}"
    echo "safety-rank sched: ${POSTTRAJ_JOINT_SAFETY_RANK_WEIGHT_SCHEDULE}"
    echo "safe-rank sched  : ${POSTTRAJ_JOINT_SAFE_FINAL_RANK_WEIGHT_SCHEDULE}"
  fi
else
  echo "counterfactual   : interface present, Stage A loss disabled"
fi
echo "============================================================"

CMD=(
  python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training.py"
  "agent=drive_jepa_perception_based_agent"
  "+agent.config.vjepa_version=2.1"
  "+agent.config.pretrain_pt_path=${PRETRAIN_PT_PATH}"
  "+agent.config.image_architecture=vjepa2_1_vit_large_384"
  "+agent.config.freeze_encoder=True"
  "+agent.config.use_lora=True"
  "+agent.config.lora_rank=${LORA_RANK}"
  "+agent.config.proposal_num=${PROPOSAL_NUM}"
  "agent.checkpoint_path=${CHECKPOINT_PATH}"
  "agent.config.posttraj_future_enabled=True"
  "agent.config.posttraj_future_loss_weight=${POSTTRAJ_FUTURE_LOSS_WEIGHT}"
  "agent.config.posttraj_future_ema_decay=0.99925"
  "agent.config.posttraj_future_frame_offsets=${POSTTRAJ_FUTURE_FRAME_OFFSETS}"
  "agent.config.posttraj_full_candidate_latent=${POSTTRAJ_FULL_CANDIDATE_LATENT}"
  "agent.config.posttraj_future_candidate_layers=2"
  "agent.config.posttraj_future_full_decoder_layers=2"
  "agent.config.posttraj_scorer_temporal_layers=1"
  "agent.config.posttraj_future_layernorm_target=True"
  "agent.config.posttraj_detach_trajectory_inputs=True"
  "agent.config.posttraj_stage2_enabled=${POSTTRAJ_STAGE2_ENABLED}"
  "agent.config.posttraj_stage2_anchor_cache_root=${POSTTRAJ_STAGE2_ANCHOR_CACHE_ROOT}"
  "agent.config.posttraj_stage2_num_local_hard=${POSTTRAJ_STAGE2_NUM_LOCAL_HARD}"
  "agent.config.posttraj_stage2_num_balanced=${POSTTRAJ_STAGE2_NUM_BALANCED}"
  "agent.config.posttraj_stage2_anchor_loss_weight=${POSTTRAJ_STAGE2_ANCHOR_LOSS_WEIGHT}"
  "agent.config.posttraj_stage2_use_anchors=${POSTTRAJ_STAGE2_USE_ANCHORS}"
  "agent.config.posttraj_stage2_sampling_profile=${POSTTRAJ_STAGE2_SAMPLING_PROFILE}"
  "agent.config.posttraj_final_topk_rank_weight=${POSTTRAJ_FINAL_TOPK_RANK_WEIGHT}"
  "agent.config.posttraj_final_topk_rank_k=${POSTTRAJ_FINAL_TOPK_RANK_K}"
  "agent.config.posttraj_final_topk_rank_score_gap=${POSTTRAJ_FINAL_TOPK_RANK_SCORE_GAP}"
  "agent.config.posttraj_final_topk_rank_margin_cap=${POSTTRAJ_FINAL_TOPK_RANK_MARGIN_CAP}"
  "agent.config.posttraj_final_topk_rank_false_topk_weight=${POSTTRAJ_FINAL_TOPK_RANK_FALSE_TOPK_WEIGHT}"
  "agent.config.posttraj_final_topk_rank_safety_weight=${POSTTRAJ_FINAL_TOPK_RANK_SAFETY_WEIGHT}"
  "agent.config.posttraj_final_topk_rank_safety_threshold=${POSTTRAJ_FINAL_TOPK_RANK_SAFETY_THRESHOLD}"
  "agent.config.posttraj_safety_hard_enabled=${POSTTRAJ_SAFETY_HARD_ENABLED}"
  "agent.config.posttraj_safety_rank_weight=${POSTTRAJ_SAFETY_RANK_WEIGHT}"
  "agent.config.posttraj_safety_threshold=${POSTTRAJ_SAFETY_THRESHOLD}"
  "agent.config.posttraj_safety_rank_topk=${POSTTRAJ_SAFETY_RANK_TOPK}"
  "agent.config.posttraj_safety_rank_margin=${POSTTRAJ_SAFETY_RANK_MARGIN}"
  "agent.config.posttraj_safety_unsafe_final_weight=${POSTTRAJ_SAFETY_UNSAFE_FINAL_WEIGHT}"
  "agent.config.posttraj_safe_final_rank_weight=${POSTTRAJ_SAFE_FINAL_RANK_WEIGHT}"
  "agent.config.posttraj_safe_final_rank_topk=${POSTTRAJ_SAFE_FINAL_RANK_TOPK}"
  "agent.config.posttraj_safe_final_rank_score_gap=${POSTTRAJ_SAFE_FINAL_RANK_SCORE_GAP}"
  "agent.config.posttraj_safe_final_rank_margin_cap=${POSTTRAJ_SAFE_FINAL_RANK_MARGIN_CAP}"
  "agent.config.posttraj_stage2_schedule_enabled=${POSTTRAJ_STAGE2_SCHEDULE_ENABLED}"
  "agent.config.posttraj_stage2_unsafe_final_weight_schedule=${POSTTRAJ_STAGE2_UNSAFE_FINAL_WEIGHT_SCHEDULE}"
  "agent.config.posttraj_stage2_safety_rank_weight_schedule=${POSTTRAJ_STAGE2_SAFETY_RANK_WEIGHT_SCHEDULE}"
  "agent.config.posttraj_stage2_safe_final_rank_weight_schedule=${POSTTRAJ_STAGE2_SAFE_FINAL_RANK_WEIGHT_SCHEDULE}"
  "agent.config.posttraj_stage2_lr_schedule=${POSTTRAJ_STAGE2_LR_SCHEDULE}"
  "agent.config.posttraj_joint_enabled=${POSTTRAJ_JOINT_ENABLED}"
  "agent.config.posttraj_joint_warmup_epochs=${POSTTRAJ_JOINT_WARMUP_EPOCHS}"
  "agent.config.posttraj_joint_ramp_epochs=${POSTTRAJ_JOINT_RAMP_EPOCHS}"
  "agent.config.posttraj_joint_anchor_loss_weight=${POSTTRAJ_JOINT_ANCHOR_LOSS_WEIGHT}"
  "agent.config.posttraj_joint_safety_schedule_enabled=${POSTTRAJ_JOINT_SAFETY_SCHEDULE_ENABLED}"
  "agent.config.posttraj_joint_unsafe_final_weight_schedule=${POSTTRAJ_JOINT_UNSAFE_FINAL_WEIGHT_SCHEDULE}"
  "agent.config.posttraj_joint_safety_rank_weight_schedule=${POSTTRAJ_JOINT_SAFETY_RANK_WEIGHT_SCHEDULE}"
  "agent.config.posttraj_joint_safe_final_rank_weight_schedule=${POSTTRAJ_JOINT_SAFE_FINAL_RANK_WEIGHT_SCHEDULE}"
  "agent.config.anchor_trajectory_path=${ANCHOR_TRAJECTORY_PATH}"
  "+agent.config.sub_score_weight=0"
  "+agent.config.final_score_weight=1"
  "agent.lr=${LR}"
  "experiment_name=${EXPERIMENT_NAME}"
  "train_test_split=navtrain"
  "split=trainval"
  "dataloader.params.batch_size=${BATCH_SIZE}"
  "dataloader.params.num_workers=${DATALOADER_NUM_WORKERS}"
  "cache_path=${CACHE_PATH}"
  "use_cache_without_dataset=True"
  "force_cache_computation=False"
  "trainer.params.max_epochs=${MAX_EPOCHS}"
  "trainer.params.accelerator=gpu"
  "trainer.params.strategy=${TRAINER_STRATEGY}"
  "trainer.params.precision=${TRAINER_PRECISION}"
  "+trainer.params.devices=${NUM_GPUS}"
)

if [[ "${DATALOADER_NUM_WORKERS}" -gt 0 ]]; then
  CMD+=("dataloader.params.prefetch_factor=${DATALOADER_PREFETCH_FACTOR}")
else
  CMD+=("dataloader.params.prefetch_factor=null")
fi
[[ -n "${LIMIT_TRAIN_BATCHES}" ]] && \
  CMD+=("trainer.params.limit_train_batches=${LIMIT_TRAIN_BATCHES}")
[[ -n "${LIMIT_VAL_BATCHES}" ]] && \
  CMD+=("trainer.params.limit_val_batches=${LIMIT_VAL_BATCHES}")
[[ -n "${CHECKPOINT_SAVE_TOP_K}" ]] && \
  CMD+=("trainer.checkpoint.save_top_k=${CHECKPOINT_SAVE_TOP_K}")
[[ -n "${CHECKPOINT_SAVE_LAST}" ]] && \
  CMD+=("+trainer.checkpoint.save_last=${CHECKPOINT_SAVE_LAST}")
[[ -n "${CHECKPOINT_MONITOR}" ]] && \
  CMD+=("trainer.checkpoint.monitor=${CHECKPOINT_MONITOR}")
[[ -n "${CHECKPOINT_MODE}" ]] && \
  CMD+=("trainer.checkpoint.mode=${CHECKPOINT_MODE}")
[[ -n "${CHECKPOINT_FILENAME}" ]] && \
  CMD+=("trainer.checkpoint.filename=${CHECKPOINT_FILENAME}")
if [[ -n "${LOG_NAME_OVERRIDE}" ]]; then
  CMD+=(
    "train_logs=['${LOG_NAME_OVERRIDE}']"
    "val_logs=['${LOG_NAME_OVERRIDE}']"
  )
fi

printf '+ %q' "${CMD[@]}"
echo
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "DRY-RUN: command construction passed"
else
  # Replace this shell: do not resume reading a shared script after a long run.
  exec "${CMD[@]}"
fi
