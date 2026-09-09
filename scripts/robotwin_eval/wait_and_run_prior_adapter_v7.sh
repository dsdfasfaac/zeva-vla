#!/usr/bin/env bash
set -euo pipefail

# One-shot continuation for the corrected v7 experiment.  The lock makes it
# safe for a human or heartbeat to inspect/reinvoke without creating duplicate
# closed-loop evaluations.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
train_root=${TRAIN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-prior-only-v7-corrected/zeva}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-prior-only-v7}
training_pid_file=${TRAINING_PID_FILE:-$train_root/launcher.pid}
state=$eval_root/pipeline_state.json
lock=$eval_root/pipeline.lock
mkdir -p "$eval_root"

if ! mkdir "$lock" 2>/dev/null; then
  echo "v7 continuation lock already exists: $lock" >&2
  exit 2
fi
cleanup() {
  rmdir "$lock" 2>/dev/null || true
}
trap cleanup EXIT

write_state() {
  local phase=$1
  local detail=$2
  python3 - "$state" "$phase" "$detail" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

destination, phase, detail = sys.argv[1:]
payload = {"schema": "zeva-robotwin-v7-pipeline-state-v1",
           "phase": phase, "detail": detail,
           "updated_at": datetime.now(timezone.utc).isoformat()}
path = Path(destination)
temporary = path.with_name(path.name + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

test -s "$training_pid_file"
training_pid=$(<"$training_pid_file")
if ! [[ "$training_pid" =~ ^[0-9]+$ ]]; then
  echo "invalid training pid: $training_pid" >&2
  exit 2
fi
if kill -0 "$training_pid" 2>/dev/null; then
  training_command=$(ps -p "$training_pid" -o args=)
  if [[ "$training_command" != *"train_robotwin_advantage10_prior_only_v7.sh"* ]]; then
    echo "pid $training_pid does not belong to the corrected v7 launcher" >&2
    exit 2
  fi
fi

write_state waiting_training "launcher_pid=$training_pid"
while kill -0 "$training_pid" 2>/dev/null; do
  sleep 30
done
if [[ ! -s "$train_root/COMPLETE" ]]; then
  write_state training_failed "launcher_pid=$training_pid exited without COMPLETE"
  exit 1
fi

write_state selecting_checkpoint "train95_validation5_only"
python3 "$zeva_root/scripts/select_robotwin_prior_adapter_checkpoint.py" "$train_root"

write_state running_fresh_validation "start_seed=5000,6000"
EVAL_ROOT="$eval_root" \
  bash "$zeva_root/scripts/robotwin_eval/launch_prior_adapter_validation_v7.sh"

validation_accepted=$(python3 - "$eval_root/validation_summary.json" <<'PY'
import json
import sys
print(str(bool(json.load(open(sys.argv[1]))["accepted_for_fresh_final_test"])).lower())
PY
)
if [[ "$validation_accepted" != true ]]; then
  write_state validation_rejected "diagnose_and_retrain"
  exit 3
fi

write_state running_fresh_final "start_seed=10000"
EVAL_ROOT="$eval_root" \
  bash "$zeva_root/scripts/robotwin_eval/launch_prior_adapter_fresh_final_v7.sh"
write_state complete "fresh final accepted and independently audited"
