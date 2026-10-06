#!/usr/bin/env bash

set -euo pipefail

CONDA_SH=${CONDA_SH:-/home/myuser/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-bench2drive}
INSTALL_ROOT=${INSTALL_ROOT:-/mnt/c2-worldmodel/2639639/Bench2Drive}
DOWNLOAD_ROOT=${DOWNLOAD_ROOT:-${INSTALL_ROOT}/downloads}
CARLA_ROOT=${CARLA_ROOT:-${INSTALL_ROOT}/CARLA_0.9.15}

CARLA_ARCHIVE="${DOWNLOAD_ROOT}/CARLA_0.9.15.tar.gz"
MAPS_ARCHIVE="${DOWNLOAD_ROOT}/AdditionalMaps_0.9.15.tar.gz"
CARLA_SIZE=8386636048
MAPS_SIZE=7375946087

CARLA_URL=${CARLA_URL:-https://carla-releases.b-cdn.net/Linux/CARLA_0.9.15.tar.gz}
MAPS_URL=${MAPS_URL:-https://carla-releases.b-cdn.net/Linux/AdditionalMaps_0.9.15.tar.gz}

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

command -v aria2c >/dev/null 2>&1 || {
  echo "ERROR: aria2c is required in ${CONDA_ENV}." >&2
  exit 2
}

mkdir -p "${DOWNLOAD_ROOT}" "${CARLA_ROOT}"
unset all_proxy ALL_PROXY

download() {
  local url=$1
  local output=$2

  until aria2c \
      --continue=true \
      --max-connection-per-server=16 \
      --split=16 \
      --min-split-size=16M \
      --file-allocation=none \
      --max-tries=0 \
      --retry-wait=5 \
      --summary-interval=30 \
      --console-log-level=warn \
      --dir="${DOWNLOAD_ROOT}" \
      --out="${output}" \
      "${url}"; do
    echo "Download interrupted for ${output}; retrying in 10 seconds." >&2
    sleep 10
  done
}

if [[ -f "${CARLA_ARCHIVE}.aria2" ]] ||
   [[ "$(stat -c %s "${CARLA_ARCHIVE}" 2>/dev/null || echo 0)" -ne "${CARLA_SIZE}" ]]; then
  download "${CARLA_URL}" "$(basename "${CARLA_ARCHIVE}")" &
  carla_download_pid=$!
else
  carla_download_pid=
fi

if [[ -f "${MAPS_ARCHIVE}.aria2" ]] ||
   [[ "$(stat -c %s "${MAPS_ARCHIVE}" 2>/dev/null || echo 0)" -ne "${MAPS_SIZE}" ]]; then
  download "${MAPS_URL}" "$(basename "${MAPS_ARCHIVE}")" &
  maps_download_pid=$!
else
  maps_download_pid=
fi

[[ -z "${carla_download_pid}" ]] || wait "${carla_download_pid}"
[[ -z "${maps_download_pid}" ]] || wait "${maps_download_pid}"

[[ ! -e "${CARLA_ARCHIVE}.aria2" ]] || {
  echo "ERROR: CARLA download is incomplete." >&2
  exit 1
}
[[ ! -e "${MAPS_ARCHIVE}.aria2" ]] || {
  echo "ERROR: AdditionalMaps download is incomplete." >&2
  exit 1
}

actual_carla_size="$(stat -c %s "${CARLA_ARCHIVE}")"
actual_maps_size="$(stat -c %s "${MAPS_ARCHIVE}")"
[[ "${actual_carla_size}" -eq "${CARLA_SIZE}" ]] || {
  echo "ERROR: CARLA archive size ${actual_carla_size}, expected ${CARLA_SIZE}." >&2
  exit 1
}
[[ "${actual_maps_size}" -eq "${MAPS_SIZE}" ]] || {
  echo "ERROR: AdditionalMaps archive size ${actual_maps_size}, expected ${MAPS_SIZE}." >&2
  exit 1
}

if [[ ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
  echo "Extracting CARLA into ${CARLA_ROOT}"
  tar -xzf "${CARLA_ARCHIVE}" -C "${CARLA_ROOT}"
fi

if [[ ! -f "${CARLA_ROOT}/.additional_maps_0.9.15_imported" ]]; then
  echo "Importing CARLA AdditionalMaps"
  mkdir -p "${CARLA_ROOT}/Import"
  cp -f "${MAPS_ARCHIVE}" "${CARLA_ROOT}/Import/AdditionalMaps_0.9.15.tar.gz"
  (
    cd "${CARLA_ROOT}"
    bash ImportAssets.sh
  )
  touch "${CARLA_ROOT}/.additional_maps_0.9.15_imported"
fi

chmod +x "${CARLA_ROOT}/CarlaUE4.sh"

echo "CARLA 0.9.15 installed at ${CARLA_ROOT}"
