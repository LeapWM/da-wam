#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B2D_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  # shellcheck source=../activate_bench2drive.sh
  source "${B2D_ROOT}/activate_bench2drive.sh"
else
  # shellcheck source=../env.sh
  source "${B2D_ROOT}/env.sh"
fi

if [[ ! -x "${CARLA_SERVER}" ]]; then
  echo "ERROR: CARLA server is not executable: ${CARLA_SERVER}" >&2
  exit 2
fi

carla_preload="${LD_PRELOAD:-}"
if [[ "$(id -u)" == "0" ]]; then
  build_dir="${TMPDIR:-/tmp}/bench2drive-carla-uid-compat"
  shim="${build_dir}/libcarla_uid_compat.so"
  mkdir -p "${build_dir}"

  if [[ ! -f "${shim}" || "${SCRIPT_DIR}/carla_uid_compat.c" -nt "${shim}" ]]; then
    gcc -shared -fPIC -O2 \
      -o "${shim}" "${SCRIPT_DIR}/carla_uid_compat.c"
  fi

  carla_preload="${shim}${carla_preload:+:${carla_preload}}"
  echo "CARLA UID compatibility: task user is UID 0; reporting a non-root UID to Unreal."
fi

carla_home="${B2D_CARLA_HOME:-${TMPDIR:-/tmp}/bench2drive-carla-home}"
mkdir -p "${carla_home}"

vulkan_root="${B2D_ROOT}/third_party/vulkan-loader/root"
carla_library_path="${CARLA_LD_LIBRARY_PATH}"
vulkan_icd="${VK_ICD_FILENAMES:-}"

host_driver_version="$(
  { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true; } \
    | sed -n '1p' \
    | tr -d '[:space:]'
)"
graphics_root="${B2D_ROOT}/third_party/nvidia-${host_driver_version}/nvidia_driver-linux-x86_64-${host_driver_version}-archive"
if [[ -n "${host_driver_version}" && \
      -f "${graphics_root}/etc/nvidia_icd.json" && \
      -f "${vulkan_root}/usr/lib/x86_64-linux-gnu/libvulkan.so.1" ]]; then
  carla_library_path="${vulkan_root}/usr/lib/x86_64-linux-gnu:${graphics_root}/lib:${carla_library_path}"
  vulkan_icd="${graphics_root}/etc/nvidia_icd.json"
  echo "CARLA Vulkan runtime: NVIDIA ${host_driver_version} user-space libraries enabled."
elif [[ -n "${host_driver_version}" ]]; then
  echo "CARLA Vulkan runtime: no local libraries matching host driver ${host_driver_version}."
fi

exec env \
  HOME="${carla_home}" \
  PATH="${SCRIPT_DIR}:${PATH}" \
  LD_LIBRARY_PATH="${carla_library_path}" \
  LD_PRELOAD="${carla_preload}" \
  VK_ICD_FILENAMES="${vulkan_icd}" \
  "${CARLA_SERVER}" \
  -SaveToUserDir \
  "$@"
