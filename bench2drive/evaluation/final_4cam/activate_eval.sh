export CONDA_PREFIX="${B2D_EVAL_ENV:-/tmp/b2d_road_env_20260921}"
export CONDA_DEFAULT_ENV=native256_eval_py38
export PATH="${CONDA_PREFIX}/bin:${PATH}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../Bench2Drive" && pwd)/env.sh"
