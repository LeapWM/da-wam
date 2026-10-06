#!/usr/bin/env bash
# Independent temporal reranking experiment; existing evaluation scripts unchanged.
set -euo pipefail
NAVSIM_V2_ROOT="${NAVSIM_V2_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-/home/myuser/miniconda3/envs/drive-jepa}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/c2-worldmodel/training_data/OpenScene/dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/c2-worldmodel/2639639/navsim_exp}"
export NUPLAN_MAPS_ROOT="${OPENSCENE_DATA_ROOT}/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_DEVKIT_ROOT="${NAVSIM_V2_ROOT}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${NAVSIM_V2_ROOT}:${NAVSIM_V2_ROOT}/vjepa2:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_ROOT}/lib/python3.9/site-packages/torch/lib:${CONDA_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export NAVSIM_AGENT_CUDA=1 NAVSIM_GPU_WORKER_AFFINITY=1
SPLIT="${SPLIT:-navtest}"
case "${SPLIT}" in
  navtest)
    ENTRY=run_pdm_score_one_stage.py
    CACHE_DEFAULT="${NAVSIM_EXP_ROOT}/Drive-JEPA-cache/metric_cache_v2"
    ;;
  navhard_two_stage)
    ENTRY=run_pdm_score.py
    CACHE_DEFAULT="${NAVSIM_EXP_ROOT}/navhard_two_stage_metric_cache_v22"
    ;;
  *) echo "Unsupported split: ${SPLIT}" >&2; exit 1 ;;
esac
MODE="${MODE:-bounded}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/tmp/posttraj_v1_e1_zeroshot_v2.ckpt}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${CACHE_DEFAULT}}"
[[ -s "${CHECKPOINT_PATH}" && -d "${METRIC_CACHE_PATH}" ]]
cd "${NAVSIM_V2_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU:-0,1}" "${CONDA_ROOT}/bin/python" -u \
  "navsim/planning/script/${ENTRY}" --config-name=posttraj_temporal_v22 \
  "train_test_split=${SPLIT}" agent=drive_jepa_posttraj_v2_agent \
  "agent.checkpoint_path=${CHECKPOINT_PATH}" "metric_cache_path=${METRIC_CACHE_PATH}" \
  worker=single_machine_thread_pool worker.use_process_pool=true \
  "worker.max_workers=${WORKERS:-16}" \
  "experiment_name=${EXPERIMENT_NAME:-posttraj_v1_${SPLIT}_temporal_${MODE}_v22}" \
  "temporal_rerank.mode=${MODE}" "$@"
