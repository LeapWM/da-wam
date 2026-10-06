#!/usr/bin/env bash
DA_WAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DA_WAM_ROOT
export MODEL_ENV="${MODEL_ENV:-/home/myuser/miniconda3/envs/drive-jepa}"
export PATH="${MODEL_ENV}/bin:${PATH}" CONDA_PREFIX="${MODEL_ENV}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${MODEL_ENV}/lib/python3.9/site-packages/torch/lib:${MODEL_ENV}/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/compat/lib"
unset PYTHONHOME LD_PRELOAD
export NAVSIM_DEVKIT_ROOT="${DA_WAM_ROOT}/navsim_v1"
export PROJECT_ROOT="${DA_WAM_ROOT}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${DA_WAM_ROOT}/third_party/vjepa2"
