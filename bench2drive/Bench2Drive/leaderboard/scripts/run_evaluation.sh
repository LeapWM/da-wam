#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B2D_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=../../env.sh
source "${B2D_ROOT}/env.sh"

# Keep the Python 3.8 evaluator isolated while allowing an explicitly selected
# external model repository to provide its lightweight CARLA bridge package.
if [[ -n "${DRIVE_JEPA_MODEL_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${DRIVE_JEPA_MODEL_PYTHONPATH}:${PYTHONPATH}"
fi

export CARLA_SERVER="${CARLA_ROOT}/CarlaUE4.sh"
export PYTHONPATH="${B2D_ROOT}/leaderboard/team_code:${PYTHONPATH}"
export SCENARIO_RUNNER_ROOT="${B2D_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${B2D_ROOT}/leaderboard"
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=$3
export PLANNER_TYPE=$9
export GPU_RANK=${10}

# TCP evaluation
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8

CUDA_VISIBLE_DEVICES=${GPU_RANK} python "${B2D_EVALUATOR_SCRIPT:-${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py}" \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET:-} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH:-} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
--gpu-rank=${GPU_RANK} \
