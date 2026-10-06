#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/environment/activate_model.sh"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/c2-worldmodel/training_data/OpenScene/dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/c2-worldmodel/2639639/navsim_exp}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}" NUPLAN_MAP_VERSION=nuplan-maps-v1.0
config_only=false
for arg in "$@"; do [[ "$arg" == --cfg* ]] && config_only=true; done
if [[ "$config_only" == false ]]; then
 : "${OUTPUT_DIR:?Set OUTPUT_DIR to a new results directory}"
 [[ ! -e "$OUTPUT_DIR" ]] || { echo 'OUTPUT_DIR already exists' >&2; exit 2; }
fi
cd "$ROOT/navsim_v1"
extra=()
[[ -z "${METRIC_CACHE_PATH:-}" ]] || extra+=("metric_cache_path=$METRIC_CACHE_PATH")
exec "$MODEL_ENV/bin/python" navsim/planning/script/run_pdm_score.py \
 train_test_split=navtest agent=drive_jepa_perception_based_agent \
 +agent.config.vjepa_version=2.1 "+agent.config.pretrain_pt_path=$ROOT/checkpoints/vjepa2_1_vitl.pt" \
 +agent.config.image_architecture=vjepa2_1_vit_large_384 +agent.config.freeze_encoder=True \
 +agent.config.use_lora=True +agent.config.lora_rank=32 +agent.config.proposal_num=32 \
 agent.config.posttraj_future_enabled=True agent.config.posttraj_future_loss_weight=0.1 \
 agent.config.posttraj_future_ema_decay=0.99925 'agent.config.posttraj_future_frame_offsets=[1]' \
 agent.config.posttraj_future_candidate_layers=2 agent.config.posttraj_future_full_decoder_layers=2 \
 agent.config.posttraj_scorer_temporal_layers=1 agent.config.posttraj_future_layernorm_target=True \
 agent.config.posttraj_detach_trajectory_inputs=True agent.config.posttraj_stage2_enabled=False \
 agent.config.posttraj_joint_enabled=False "agent.config.anchor_trajectory_path=$ROOT/navsim_v1/data/8192.npy" \
 +agent.config.sub_score_weight=0 +agent.config.final_score_weight=1 \
 "agent.checkpoint_path=${CHECKPOINT_PATH:-$ROOT/checkpoints/navsim_final.ckpt}" \
 experiment_name=da_wam_final_v1 "output_dir=${OUTPUT_DIR:-/tmp/da_wam_v1_config_only}" \
 worker=single_machine_thread_pool "worker.max_workers=${WORKERS:-4}" worker.use_process_pool=true \
 "${extra[@]}" "$@"
