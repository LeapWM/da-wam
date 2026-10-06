#!/usr/bin/env bash
set -euo pipefail
entry="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
result="${DRIVE_JEPA_B2D_RESULT_DIR:?}"
config="${DRIVE_JEPA_B2D_CONFIG:?}"
mkdir -p "$result/sensor_output"
CUDA_VISIBLE_DEVICES="${DRIVE_JEPA_GPU}" bash "$entry/sources/4cam/run_model.sh" "$config" >"$result/model_server.log" 2>&1 &
model_pid=$!
cleanup() { kill "$model_pid" 2>/dev/null || true; wait "$model_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for attempt in $(seq 1 600); do
 if grep -q 'bridge listening' "$result/model_server.log"; then break; fi
 if ! kill -0 "$model_pid" 2>/dev/null; then tail -n 60 "$result/model_server.log"; exit 1; fi
 sleep 1
done
grep -q 'bridge listening' "$result/model_server.log"
source "$entry/activate_eval.sh"
export DRIVE_JEPA_MODEL_PYTHONPATH="$entry:$entry/sources/4cam:/dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/dependencies/Drive-JEPA-V2-EMA-JQTF"
export B2D_EVALUATOR_SCRIPT="$entry/leaderboard_evaluator.py"
export B2D_CLIENT_CACHE="/tmp/native256_client_cache_${CARLA_PORT}"
mkdir -p "$B2D_CLIENT_CACHE"
cd /dahuafs/userdata/2639639/Code/leap-auto-wam/bench2drive/Bench2Drive
bash leaderboard/scripts/run_evaluation.sh "$CARLA_PORT" "$TM_PORT" 1 "$entry/routes.xml" "$entry/evaluation_agent.py" "$config" "$result/results.json" "$result/sensor_output" only_ctrl "$CARLA_GPU" 2>&1 | tee "$result/evaluation.log"
