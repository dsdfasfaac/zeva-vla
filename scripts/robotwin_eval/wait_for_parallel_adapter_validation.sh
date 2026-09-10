#!/usr/bin/env bash
set -euo pipefail

# Resume the anchored-v9 validation coordinator after its two paired splits
# were deliberately continued by independent launchers.  This process never
# starts or mutates an in-flight split.  It waits for both launchers to publish
# an audited terminal report, then re-enters the normal validation/final gate.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
candidate_root=${CANDIDATE_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-anchored-v9/adapter}
base_checkpoint=${BASE_CHECKPOINT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-action-expert-v8/baseline/003000}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-anchored-v9/adapter}
split_c_name=${SPLIT_C_NAME:-split-e}
split_d_name=${SPLIT_D_NAME:-split-f}
start_seed_c=${START_SEED_C:-9000}
start_seed_d=${START_SEED_D:-12000}
excluded_validation_starts=${EXCLUDED_VALIDATION_STARTS:-5000,6000,7000,8000}
episodes=${EPISODES:-8}
poll_seconds=${POLL_SECONDS:-60}

declare -A launcher_pid_files=(
  ["$split_c_name"]="$eval_root/$split_c_name-zeva.pid"
  ["$split_d_name"]="$eval_root/$split_d_name-zeva.pid"
)

write_state() {
  printf '{"state":"%s","updated":"%s"}\n' "$1" "$(date -Iseconds)" \
    > "$eval_root/pipeline_state.json"
}

split_is_complete() {
  local split=$1
  python3 - "$eval_root/$split" "$episodes" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2]) * 10
state_path = root / "state.json"
report_path = root / "paired_report.json"
if not state_path.is_file() or not report_path.is_file():
    raise SystemExit(1)
state = json.loads(state_path.read_text())
report = json.loads(report_path.read_text())
if state.get("state") != "complete":
    raise SystemExit(1)
if (report.get("schema") != "zeva-robotwin-formal-paired-report-v1"
        or report.get("total_paired_episodes") != expected):
    raise RuntimeError(f"{root.name}: invalid terminal paired report")
PY
}

write_state waiting_for_parallel_validation_reports
while true; do
  all_complete=true
  for split in "$split_c_name" "$split_d_name"; do
    if split_is_complete "$split"; then
      continue
    fi
    all_complete=false
    pid_file=${launcher_pid_files[$split]}
    if [[ ! -s "$pid_file" ]]; then
      write_state "${split}_launcher_pid_missing"
      exit 2
    fi
    pid=$(<"$pid_file")
    if ! kill -0 "$pid" 2>/dev/null; then
      write_state "${split}_launcher_failed_before_terminal_report"
      exit 3
    fi
  done
  [[ "$all_complete" == true ]] && break
  sleep "$poll_seconds"
done

write_state resuming_preregistered_validation_gate
CANDIDATE_ROOT="$candidate_root" BASE_CHECKPOINT="$base_checkpoint" \
  EVAL_ROOT="$eval_root" SPLIT_C_NAME="$split_c_name" SPLIT_D_NAME="$split_d_name" \
  START_SEED_C="$start_seed_c" START_SEED_D="$start_seed_d" \
  EPISODES="$episodes" \
  EXCLUDED_VALIDATION_STARTS="$excluded_validation_starts" \
  bash "$zeva_root/scripts/robotwin_eval/wait_and_run_anchored_v9.sh"
