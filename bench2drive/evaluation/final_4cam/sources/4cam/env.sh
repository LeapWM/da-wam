#!/usr/bin/env bash
set -euo pipefail
entry_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/Drive-JEPA-V2-EMA-JQTF/bench2drive/scripts/activate_model_env.sh
export PYTHONPATH="${entry_dir}:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/geometry/src:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/dense_data/src:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/Drive-JEPA-V2-EMA-JQTF:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/Drive-JEPA-V2-EMA-JQTF/navsim_v1:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/Drive-JEPA-V2-EMA-JQTF/navsim_v1/vjepa2:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/data_support/src"
export B2D_NEW_CACHE="${B2D_NEW_CACHE:-/mnt/c2-worldmodel/2639639/Bench2Drive/sparsedrivev2_base_10hz}"
export B2D_DENSE_INDEX="${B2D_DENSE_INDEX:-/mnt/c2-worldmodel/2639639/Bench2Drive/ema_jqtf_10hz}"
export JinnTrainResult="${JinnTrainResult:-${NAVSIM_EXP_ROOT:-${B2D_DENSE_INDEX}/runs}}"
export NAVSIM_EXP_ROOT="${JinnTrainResult}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME LD_PRELOAD
