#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B2D_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
driver_version="${B2D_NVIDIA_DRIVER_VERSION:-$(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    | sed -n '1p' \
    | tr -d '[:space:]'
)}"
if [[ -z "${driver_version}" ]]; then
  echo "ERROR: cannot determine the NVIDIA driver version." >&2
  exit 2
fi

graphics_root="${B2D_ROOT}/third_party/nvidia-${driver_version}/nvidia_driver-linux-x86_64-${driver_version}-archive"
vulkan_root="${B2D_ROOT}/third_party/vulkan-loader/root"

loader_lib="${vulkan_root}/usr/lib/x86_64-linux-gnu"
driver_lib="${graphics_root}/lib"
vulkaninfo="${vulkan_root}/usr/bin/vulkaninfo"
icd="${graphics_root}/etc/nvidia_icd.json"

for required in "${loader_lib}/libvulkan.so.1" "${driver_lib}/libGLX_nvidia.so.0" "${vulkaninfo}" "${icd}"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: missing Vulkan runtime file: ${required}" >&2
    exit 2
  fi
done

exec env \
  LD_LIBRARY_PATH="${loader_lib}:${driver_lib}:${LD_LIBRARY_PATH:-}" \
  VK_ICD_FILENAMES="${icd}" \
  "${vulkaninfo}" --summary
