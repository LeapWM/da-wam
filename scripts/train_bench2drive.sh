#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/bench2drive/evaluation/final_4cam/sources/4cam/train_bench2drive.sh" "$@"
