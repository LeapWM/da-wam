#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${DRIVE_JEPA_B2D_RESULT_DIR:?DRIVE_JEPA_B2D_RESULT_DIR must be set}"
RESULT_JSON="${RESULT_DIR}/results.json"
ATTEMPT_DIR="${RESULT_DIR}/attempts"
MAX_ATTEMPTS="${B2D_MAX_RESUME_ATTEMPTS:-20}"
MAX_STALLED_ATTEMPTS="${B2D_MAX_STALLED_ATTEMPTS:-3}"
RETRY_DELAY_SECONDS="${B2D_RETRY_DELAY_SECONDS:-15}"

if [[ ! "${MAX_ATTEMPTS}" =~ ^[0-9]+$ ]] || (( MAX_ATTEMPTS < 1 )); then
  echo "ERROR: B2D_MAX_RESUME_ATTEMPTS must be a positive integer." >&2
  exit 2
fi
if [[ ! "${MAX_STALLED_ATTEMPTS}" =~ ^[0-9]+$ ]] || (( MAX_STALLED_ATTEMPTS < 1 )); then
  echo "ERROR: B2D_MAX_STALLED_ATTEMPTS must be a positive integer." >&2
  exit 2
fi

mkdir -p "${ATTEMPT_DIR}"

read_state() {
  if [[ ! -s "${RESULT_JSON}" ]]; then
    printf '%s\t%s\t%s\n' "Missing" "0" "0"
    return
  fi
  jq -r '[.entry_status // "Missing", ._checkpoint.progress[0] // 0, ._checkpoint.progress[1] // 0] | @tsv' "${RESULT_JSON}"
}

archive_logs() {
  local label="$1"
  local source
  for source in evaluation.log model_server.log; do
    if [[ -s "${RESULT_DIR}/${source}" ]]; then
      cp -p "${RESULT_DIR}/${source}" "${ATTEMPT_DIR}/${label}.${source}"
    fi
  done
}

port_is_listening() {
  local port="$1"
  ss -ltn | grep -qE ":${port}[[:space:]]"
}

wait_for_ports() {
  local ports=(
    "${CARLA_PORT:-2000}"
    "$(( ${CARLA_PORT:-2000} + 1 ))"
    "${TM_PORT:-8000}"
    "${DRIVE_JEPA_BRIDGE_PORT:-50123}"
  )
  local wait_index
  local busy
  for wait_index in $(seq 1 60); do
    busy=0
    for port in "${ports[@]}"; do
      if port_is_listening "${port}"; then
        busy=1
        break
      fi
    done
    if (( busy == 0 )); then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: worker ports did not become free: ${ports[*]}" >&2
  return 1
}

timestamp="$(date '+%Y%m%d_%H%M%S')"
archive_logs "before_resume_${timestamp}"

stalled_attempts=0
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  IFS=$'\t' read -r entry_status progress total < <(read_state)
  echo "Resume attempt ${attempt}/${MAX_ATTEMPTS}: entry_status=${entry_status}, progress=${progress}/${total}"
  if [[ "${entry_status}" == "Finished" ]] && (( total > 0 && progress == total )); then
    echo "Shard is already complete."
    exit 0
  fi

  wait_for_ports
  attempt_started="$(date '+%Y%m%d_%H%M%S')"
  set +e
  bash "${SCRIPT_DIR}/run_benchmark.sh"
  run_status=$?
  set -e
  attempt_label="attempt_$(printf '%02d' "${attempt}")_${attempt_started}_rc${run_status}"
  archive_logs "${attempt_label}"

  IFS=$'\t' read -r next_entry_status next_progress next_total < <(read_state)
  echo "Attempt ${attempt} ended: status=${run_status}, entry_status=${next_entry_status}, progress=${next_progress}/${next_total}"
  if [[ "${next_entry_status}" == "Finished" ]] && (( next_total > 0 && next_progress == next_total )); then
    echo "Shard completed after resume attempt ${attempt}."
    exit 0
  fi

  if (( next_progress > progress )); then
    stalled_attempts=0
  else
    stalled_attempts=$((stalled_attempts + 1))
  fi
  if (( stalled_attempts >= MAX_STALLED_ATTEMPTS )); then
    echo "ERROR: no route progress in ${stalled_attempts} consecutive attempts." >&2
    exit 1
  fi

  echo "CARLA/evaluator stopped before shard completion; retrying in ${RETRY_DELAY_SECONDS}s."
  sleep "${RETRY_DELAY_SECONDS}"
done

echo "ERROR: exhausted ${MAX_ATTEMPTS} resume attempts." >&2
exit 1
