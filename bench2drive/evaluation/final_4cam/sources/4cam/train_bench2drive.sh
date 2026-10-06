#!/usr/bin/env bash
set -euo pipefail
entry_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/Drive-JEPA-V2-EMA-JQTF/bench2drive/scripts/activate_model_env.sh
export PYTHONPATH="${entry_dir}:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/geometry/src:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/dense_data/src:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/Drive-JEPA-V2-EMA-JQTF:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/Drive-JEPA-V2-EMA-JQTF/navsim_v1:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/Drive-JEPA-V2-EMA-JQTF/navsim_v1/vjepa2:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/data_support/src"
export B2D_NEW_CACHE="${B2D_NEW_CACHE:-/mnt/c2-worldmodel/2639639/Bench2Drive/sparsedrivev2_base_10hz}"
export B2D_DENSE_INDEX="${B2D_DENSE_INDEX:-/mnt/c2-worldmodel/2639639/Bench2Drive/ema_jqtf_10hz}"
export B2D_ROAD_ENV="${B2D_ROAD_ENV:-/mnt/c2-worldmodel/2639639/Bench2Drive/envs/road_py38_carla0915}"
export B2D_CACHE_BUNDLE="${B2D_CACHE_BUNDLE:-${B2D_DENSE_INDEX}/bundles/traffic_3s_scalar_speed_v6}"
export JinnTrainResult="${JinnTrainResult:-${NAVSIM_EXP_ROOT:-${B2D_DENSE_INDEX}/runs}}"
export NAVSIM_EXP_ROOT="${JinnTrainResult}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1 TBB_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-PostTraj_TrafficScore_Native256_4Cam_FrontFuture_10Hz}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME LD_PRELOAD
if [[ ! -x "${B2D_ROAD_ENV}/bin/python" ]]; then
  echo "Missing Bench2Drive road-label Python: ${B2D_ROAD_ENV}/bin/python" >&2
  exit 2
fi
if [[ "${DRY_RUN:-false}" == true ]]; then
  echo "PostTraj TrafficScore native NAVSIM 256D future; 4 cameras, front future only; official 950/50; 247656 observed 10Hz frames; 3s x 6"
  echo "NAVSIM_EXP_ROOT   : ${NAVSIM_EXP_ROOT}"
  echo "Num GPUs (DDP)    : ${NUM_GPUS:-8}"
  echo "Batch size / GPU  : ${BATCH_SIZE:-8}"
  echo "Global batch      : $(( ${NUM_GPUS:-8} * ${BATCH_SIZE:-8} * ${ACCUMULATE_GRAD_BATCHES:-1} ))"
  echo "trainer.params.precision=bf16-mixed"
  echo "trainer.params.strategy=${TRAINER_STRATEGY:-ddp_find_unused_parameters_true}"
  echo "+trainer.params.devices=${NUM_GPUS:-8}"
  echo "Ego status        : pose_zero3 + scalar_speed + acceleration2 + command6 (12D)"
  echo "Future space      : projected_map 256D (native NAVSIM PostTraj)"
  echo "Scorer context    : V-JEPA LoRA map 1024D"
  echo "Optimizer         : AdamW; wd=${WEIGHT_DECAY:-0.001}; warmup=${WARMUP_STEPS:-500}; cosine min_lr=${MIN_LR:-1e-6}"
  echo "Peak LR ratios    : generator/scorer/future = 1/1/2"
  echo "Future loss       : e0-9 .1; e10-14 .075; e15-19 .05; e20-24 .025; e25-29 0"
  echo "Gradient clip     : ${GRADIENT_CLIP_VAL:-1.0}"
  echo "Checkpoint policy : best-plan(ADE+0.5*FDE), best-score(rule), last"
  echo "Validation        : skip epochs 0-9; every epoch from epoch ${VALIDATION_START_EPOCH:-10}"
  echo "Cache bundle      : ${B2D_CACHE_BUNDLE}"
  echo "Road-label Python : ${B2D_ROAD_ENV}/bin/python"
  echo "Cache generation  : ${PRECOMPUTE_FULL_CACHE:-true}; workers=${CACHE_WORKERS:-128}"
  echo "DRY-RUN: command printed above, not executing."
  exit 0
fi
if [[ "${PREPARE_ONLY:-false}" == true ]]; then exec python -B -u "/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/dense_data/src/prepare.py"; fi
if [[ ! -f "${B2D_DENSE_INDEX}/index.json" ]]; then python -B -u "/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/dense_data/src/prepare.py"; fi
if [[ "${PRECOMPUTE_FULL_CACHE:-true}" == true ]]; then
  cache_output="${CACHE_OUTPUT:-${B2D_DENSE_INDEX}/audits/full_3s_scalar_speed_128_20260922}"
  if python -B -u "${entry_dir}/check_training_ready.py" --probe; then
    echo "Reusing validated full cache bundle: ${B2D_CACHE_BUNDLE}"
  else
    python -B -u "${entry_dir}/precache_full.py" --preflight-only --output "${cache_output}"
    python -B -u "${entry_dir}/precache_full.py" --workers "${CACHE_WORKERS:-128}" --output "${cache_output}"
    python -B -u "${entry_dir}/create_cache_bundle.py"
  fi
fi
if [[ "${REQUIRE_FULL_CACHE:-true}" == true ]]; then
  python -B -u "${entry_dir}/check_training_ready.py"
fi
python -B -u "${entry_dir}/check_multiview_ready.py"
extra=()
if [[ "${SMOKE:-false}" == true ]]; then extra+=(--smoke); elif [[ "${NUM_GPUS:-8}" != 8 ]]; then echo 'Training requires 8 GPUs';exit 2; fi
exec python -B -u "${entry_dir}/train.py" --devices "${NUM_GPUS:-8}" --batch-size "${BATCH_SIZE:-8}" --workers "${DATALOADER_NUM_WORKERS:-4}" --prefetch-factor "${DATALOADER_PREFETCH_FACTOR:-4}" --epochs "${MAX_EPOCHS:-30}" --validation-start-epoch "${VALIDATION_START_EPOCH:-10}" --lr "${LR:-1e-4}" --warmup-steps "${WARMUP_STEPS:-500}" --min-lr "${MIN_LR:-1e-6}" --weight-decay "${WEIGHT_DECAY:-0.001}" --gradient-clip-val "${GRADIENT_CLIP_VAL:-1.0}" "${extra[@]}"
