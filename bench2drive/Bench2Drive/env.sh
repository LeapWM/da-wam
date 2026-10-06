#!/usr/bin/env bash
# Source this file after activating the bench2drive conda environment:
#   conda activate bench2drive
#   source /dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/Bench2Drive/env.sh

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the bench2drive conda environment first." >&2
  return 2 2>/dev/null || exit 2
fi

_B2D_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export B2D_ROOT="${B2D_ROOT:-${_B2D_SCRIPT_DIR}}"
export B2D_DATA_ROOT="${B2D_DATA_ROOT:-/mnt/c2-worldmodel/training_data/Bench2Drive}"
export CARLA_ROOT="${CARLA_ROOT:-/mnt/c2-worldmodel/2639639/Bench2Drive/CARLA_0.9.15}"
export CARLA_SERVER="${CARLA_SERVER:-${CARLA_ROOT}/CarlaUE4.sh}"
export CARLA_SERVER_LAUNCHER="${CARLA_SERVER_LAUNCHER:-${B2D_ROOT}/tools/run_carla_server.sh}"
export SCENARIO_RUNNER_ROOT="${SCENARIO_RUNNER_ROOT:-${B2D_ROOT}/scenario_runner}"
export LEADERBOARD_ROOT="${LEADERBOARD_ROOT:-${B2D_ROOT}/leaderboard}"

export PYTHONNOUSERSITE=1
unset PYTHONHOME

_B2D_PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
_B2D_TORCH_LIB="${CONDA_PREFIX}/lib/python${_B2D_PYTHON_VERSION}/site-packages/torch/lib"
if [[ -d "${_B2D_TORCH_LIB}" ]]; then
  export LD_LIBRARY_PATH="${_B2D_TORCH_LIB}:${CONDA_PREFIX}/lib"
else
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
fi

# Keep conda/Torch libraries out of the native Unreal Engine process while
# retaining the host NVIDIA driver search paths it needs for off-screen Vulkan.
export CARLA_LD_LIBRARY_PATH="${CARLA_LD_LIBRARY_PATH:-/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/compat/lib:/usr/local/cuda/lib64:/usr/lib64:/lib64}"

# Keep locally built PyTorch/CUDA extensions portable across the Factory GPU
# pools used by Bench2Drive: L20 (Ada, sm_89) and H800 (Hopper, sm_90).
if [[ -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  export CUDA_HOME="${CONDA_PREFIX}"
else
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;9.0}"

_B2D_PYTHON_PATHS=(
  "${B2D_ROOT}"
  "${CARLA_ROOT}/PythonAPI"
  "${CARLA_ROOT}/PythonAPI/carla"
  "${SCENARIO_RUNNER_ROOT}"
  "${LEADERBOARD_ROOT}"
)

_B2D_JOINED_PATH="$(IFS=:; echo "${_B2D_PYTHON_PATHS[*]}")"
# Do not inherit the host PYTHONPATH: it can reintroduce the system PyTorch.
export PYTHONPATH="${_B2D_JOINED_PATH}${B2D_EXTRA_PYTHONPATH:+:${B2D_EXTRA_PYTHONPATH}}"

unset _B2D_SCRIPT_DIR _B2D_PYTHON_VERSION _B2D_TORCH_LIB
unset _B2D_PYTHON_PATHS _B2D_JOINED_PATH
