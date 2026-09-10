#!/usr/bin/env bash
set -euo pipefail

# Durable contingency coordinator. It never reads v9 closed-loop results when
# selecting a checkpoint: training completion -> validation5-only selection ->
# two fresh disjoint closed-loop validation streams.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
candidate_root=${CANDIDATE_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-foundation-adapter-v10/adapter}
foundation=${FOUNDATION_CHECKPOINT:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-foundation-adapter-v10}
mkdir -p "$eval_root"

write_state() {
  printf '{"state":"%s","updated":"%s"}\n' "$1" "$(date -Iseconds)" \
    > "$eval_root/pipeline_state.json"
}

write_state waiting_for_training
while [[ ! -s "$candidate_root/COMPLETE" ]]; do
  pid_file=$candidate_root/launcher.pid
  if [[ -s "$pid_file" ]] && ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    write_state training_failed
    exit 2
  fi
  sleep 60
done

write_state selecting_checkpoint
if ! python3 "$zeva_root/scripts/select_robotwin_foundation_adapter_v10.py" \
  "$candidate_root" "$foundation" > "$eval_root/selection.log" 2>&1; then
  write_state offline_selection_rejected
  exit 3
fi

write_state running_closed_loop_validation
if ! TRAIN_ROOT="$candidate_root" EVAL_ROOT="$eval_root" \
  bash "$zeva_root/scripts/robotwin_eval/launch_foundation_adapter_v10_validation.sh" \
  > "$eval_root/validation.launcher.log" 2>&1; then
  if [[ -s "$eval_root/validation_summary.json" ]]; then
    write_state closed_loop_validation_rejected
  else
    write_state closed_loop_validation_failed
  fi
  exit 4
fi

write_state ready_for_fresh_final
