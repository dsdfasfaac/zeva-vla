#!/usr/bin/env bash
set -euo pipefail

# Durable coordinator: wait for both training branches, freeze a validation5-
# only shared checkpoint, run two disjoint closed-loop validation splits, and
# launch the untouched-anchor/Base/ZeVA final only after validation passes.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
pair_root=${PAIR_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-action-expert-v8}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-action-expert-v8}
mkdir -p "$eval_root"

printf '{"state":"waiting_for_training","updated":"%s"}\n' "$(date -Iseconds)" \
  > "$eval_root/pipeline_state.json"
while [[ ! -s "$pair_root/baseline/COMPLETE" || ! -s "$pair_root/zeva/COMPLETE" ]]; do
  sleep 60
done

printf '{"state":"selecting_checkpoint","updated":"%s"}\n' "$(date -Iseconds)" \
  > "$eval_root/pipeline_state.json"
if ! python3 "$zeva_root/scripts/select_robotwin_action_expert_pair_v8.py" "$pair_root" \
  > "$eval_root/selection.log" 2>&1; then
  printf '{"state":"offline_selection_rejected","updated":"%s"}\n' "$(date -Iseconds)" \
    > "$eval_root/pipeline_state.json"
  exit 3
fi

printf '{"state":"running_closed_loop_validation","updated":"%s"}\n' "$(date -Iseconds)" \
  > "$eval_root/pipeline_state.json"
if ! EVAL_ROOT="$eval_root" PAIR_ROOT="$pair_root" \
  bash "$zeva_root/scripts/robotwin_eval/launch_action_expert_validation_v8.sh" \
  > "$eval_root/validation.launcher.log" 2>&1; then
  printf '{"state":"closed_loop_validation_rejected","updated":"%s"}\n' "$(date -Iseconds)" \
    > "$eval_root/pipeline_state.json"
  exit 4
fi

printf '{"state":"running_fresh_final","updated":"%s"}\n' "$(date -Iseconds)" \
  > "$eval_root/pipeline_state.json"
EVAL_ROOT="$eval_root" \
  bash "$zeva_root/scripts/robotwin_eval/launch_action_expert_fresh_final_v8.sh" \
  > "$eval_root/final.launcher.log" 2>&1
printf '{"state":"complete","updated":"%s"}\n' "$(date -Iseconds)" \
  > "$eval_root/pipeline_state.json"
