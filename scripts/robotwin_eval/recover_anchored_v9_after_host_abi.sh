#!/usr/bin/env bash
set -euo pipefail

# One-shot recovery for the 2026-09-10 split-c host ABI failure.  The original
# aigc31 service failed before readiness because its torch 2.5.1 runtime could
# not load the torch-2.4.1 Mamba extension.  Preserve a completed split-d, prove
# split-c produced zero episodes, archive the failed attempt, then resume the
# immutable validation plan sequentially on ABI-compatible aigc32.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-anchored-v9/joint}
old_coordinator_pid=${OLD_COORDINATOR_PID:-2751996}
failed_split=$eval_root/split-c
archive=$eval_root/split-c-failed-aigc31-torch251-mamba-abi-noepisodes

while kill -0 "$old_coordinator_pid" 2>/dev/null; do
  sleep 30
done

python3 - "$eval_root/pipeline_state.json" "$eval_root/split-d/paired_report.json" \
  "$eval_root/split-d/baseline/report.json" "$failed_split/baseline/progress" <<'PY'
import json
import sys
from pathlib import Path

state_path, split_d_report, split_d_baseline_report, progress_root = map(
    Path, sys.argv[1:]
)
state = json.loads(state_path.read_text())
if state.get("state") != "closed_loop_validation_rejected":
    raise SystemExit(f"unexpected old coordinator terminal state: {state}")
if split_d_report.is_file():
    report = json.loads(split_d_report.read_text())
    if report.get("total_paired_episodes") != 80:
        raise SystemExit("split-d paired report is incomplete")
    preserved = "paired Base/ZeVA"
elif split_d_baseline_report.is_file():
    report = json.loads(split_d_baseline_report.read_text())
    if report.get("total_episodes") != 80:
        raise SystemExit("split-d Base report is incomplete")
    preserved = "Base only; ZeVA will resume under the recovered launcher"
else:
    raise SystemExit("split-d did not complete even its Base condition")
completed = 0
for path in progress_root.glob("*.json"):
    completed += len(json.loads(path.read_text()).get("episode_results", ()))
if completed:
    raise SystemExit(f"split-c produced {completed} episodes; automatic host recovery forbidden")
print(f"recovery preconditions passed: split-d {preserved}, split-c episodes=0")
PY

if [[ -e "$archive" ]]; then
  echo "recovery archive already exists: $archive" >&2
  exit 3
fi
mv "$failed_split" "$archive"
for item in split-c.pid split-c.launcher.log validation.launcher.log; do
  if [[ -e "$eval_root/$item" ]]; then
    mv "$eval_root/$item" "$archive/$item"
  fi
done

python3 - "$eval_root/runtime_recovery_trigger.json" "$old_coordinator_pid" "$archive" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

destination, old_pid, archive = sys.argv[1:]
payload = {
    "schema": "zeva-robotwin-anchored-v9-runtime-recovery-trigger-v1",
    "reason": "aigc31 torch-2.5.1 versus Mamba torch-2.4.1 ABI mismatch",
    "failed_split_completed_episodes": 0,
    "preserved_completed_split": "split-d Base or paired result, as audited by watcher",
    "recovery_model_host": "aigc32",
    "checkpoint_seed_instruction_plan_unchanged": True,
    "old_coordinator_pid": int(old_pid),
    "failed_attempt_archive": archive,
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
path = Path(destination)
temporary = path.with_name(path.name + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY

exec bash "$zeva_root/scripts/robotwin_eval/wait_and_run_anchored_v9.sh"
