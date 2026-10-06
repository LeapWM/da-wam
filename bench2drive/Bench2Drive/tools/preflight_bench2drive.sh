#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B2D_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "bench2drive" ]]; then
  echo "ERROR: expected conda environment 'bench2drive', got '${CONDA_DEFAULT_ENV:-<unset>}'." >&2
  exit 2
fi

# shellcheck source=../env.sh
source "${B2D_ROOT}/env.sh"

status=0

if [[ -d "${B2D_DATA_ROOT}" ]]; then
  data_archive_count="$(find "${B2D_DATA_ROOT}" -maxdepth 1 -type f -name '*.tar.gz' | wc -l)"
  echo "Bench2Drive data  : ${B2D_DATA_ROOT} (${data_archive_count} archives)"
else
  echo "Bench2Drive data  : MISSING - ${B2D_DATA_ROOT}"
  status=1
fi

python - <<'PY'
import os
import sys

print("Python executable :", sys.executable)
print("Python version    :", sys.version.split()[0])
print("CONDA_PREFIX      :", os.environ.get("CONDA_PREFIX"))
print("LD_LIBRARY_PATH   :", os.environ.get("LD_LIBRARY_PATH"))

import torch
print("PyTorch           :", torch.__version__)
print("PyTorch location  :", torch.__file__)
print("PyTorch CUDA      :", torch.version.cuda)
print("CUDA available    :", torch.cuda.is_available())
print("CUDA device count :", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    capability = torch.cuda.get_device_capability(index)
    print(
        f"  cuda:{index}: {torch.cuda.get_device_name(index)} "
        f"(sm_{capability[0]}{capability[1]})"
    )

for module in (
    "cv2",
    "pygame",
    "py_trees",
    "shapely",
    "networkx",
    "xmlschema",
    "srunner",
    "leaderboard",
):
    imported = __import__(module)
    print(f"{module:<18}: {getattr(imported, '__file__', '<namespace>')}")

try:
    import carla
    print("carla             :", getattr(carla, "__file__", "<egg/module>"))
except Exception as error:
    print("carla             : MISSING/FAILED -", error)
PY

if [[ ! -x "${CARLA_SERVER}" ]]; then
  echo "CARLA server      : MISSING - ${CARLA_SERVER}"
  status=1
else
  echo "CARLA server      : ${CARLA_SERVER}"
fi

if command -v vulkaninfo >/dev/null 2>&1; then
  echo "Vulkan            : available ($(command -v vulkaninfo))"
else
  echo "Vulkan            : vulkaninfo not visible in this container"
  status=1
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "NVIDIA runtime    : available"
else
  echo "NVIDIA runtime    : not visible in this container"
  status=1
fi

exit "${status}"
