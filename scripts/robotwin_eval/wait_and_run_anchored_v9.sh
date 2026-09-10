#!/usr/bin/env bash
set -euo pipefail

# Durable anchored-v9 coordinator.  A validation5-only offline gate is frozen
# after training.  Closed-loop validation and the reserved final are strictly
# downstream and cannot change the selected checkpoint.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
candidate_root=${CANDIDATE_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-anchored-v9/joint}
base_checkpoint=${BASE_CHECKPOINT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-action-expert-v8/baseline/003000}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-anchored-v9/joint}
split_c_name=${SPLIT_C_NAME:-split-c}
split_d_name=${SPLIT_D_NAME:-split-d}
start_seed_c=${START_SEED_C:-7000}
start_seed_d=${START_SEED_D:-8000}
excluded_validation_starts=${EXCLUDED_VALIDATION_STARTS:-5000,6000}
mkdir -p "$eval_root"

write_state() {
  printf '{"state":"%s","updated":"%s"}\n' "$1" "$(date -Iseconds)" \
    > "$eval_root/pipeline_state.json"
}

write_state waiting_for_training
while [[ ! -s "$candidate_root/COMPLETE" ]]; do
  if [[ -s "$candidate_root/FAILED" ]]; then
    write_state training_failed
    exit 2
  fi
  sleep 60
done

write_state selecting_checkpoint
if ! python3 "$zeva_root/scripts/select_robotwin_anchored_v9.py" \
  "$candidate_root" "$base_checkpoint" > "$eval_root/selection.log" 2>&1; then
  write_state offline_selection_rejected
  exit 3
fi

write_state running_closed_loop_validation
if ! CANDIDATE_ROOT="$candidate_root" BASE_CHECKPOINT="$base_checkpoint" \
  EVAL_ROOT="$eval_root" SPLIT_C_NAME="$split_c_name" SPLIT_D_NAME="$split_d_name" \
  START_SEED_C="$start_seed_c" START_SEED_D="$start_seed_d" \
  EXCLUDED_VALIDATION_STARTS="$excluded_validation_starts" \
  bash "$zeva_root/scripts/robotwin_eval/launch_anchored_v9_validation.sh" \
  > "$eval_root/validation.launcher.log" 2>&1; then
  write_state closed_loop_validation_rejected
  exit 4
fi

write_state running_fresh_final
EVAL_ROOT="$eval_root" VALIDATION_SPLIT_C="$split_c_name" VALIDATION_SPLIT_D="$split_d_name" \
  bash "$zeva_root/scripts/robotwin_eval/launch_anchored_v9_fresh_final.sh" \
  > "$eval_root/final.launcher.log" 2>&1
write_state complete
