#!/usr/bin/env bash
set -euo pipefail

# Stop a paired final after its Base condition is complete when the immutable
# Base floor has already failed. This preserves the complete Base report and
# frozen seed manifest while avoiding pointless downstream conditions that
# cannot change the final acceptance decision.

final_root=${FINAL_ROOT:?FINAL_ROOT is required}
paired_pid=${PAIRED_PID:?PAIRED_PID is required}
accelerator_pid=${ACCELERATOR_PID:-}
minimum_successes=${MINIMUM_SUCCESSES:-114}
expected_episodes=${EXPECTED_EPISODES:-200}
poll_seconds=${POLL_SECONDS:-1}
state_file=$final_root/base_floor_stop_state.json

write_state() {
  local state=$1
  printf '{"state":"%s","updated":"%s","paired_pid":%s}\n' \
    "$state" "$(date -Iseconds)" "$paired_pid" > "$state_file"
}

kill -0 "$paired_pid"
command=$(ps -p "$paired_pid" -o args=)
[[ "$command" == *"launch_paired_formal_eval.sh"* ]] || {
  echo "PAIRED_PID is not the paired launcher: $command" >&2
  exit 2
}

write_state waiting_for_complete_base_report
while [[ ! -s "$final_root/baseline/report.json" ]]; do
  kill -0 "$paired_pid" || {
    write_state paired_launcher_died_before_base_report
    exit 3
  }
  sleep "$poll_seconds"
done

readarray -t totals < <(python3 - "$final_root/baseline/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(report.get("total_episodes", -1))
print(report.get("total_successes", -1))
PY
)
episodes=${totals[0]}
successes=${totals[1]}
if [[ "$episodes" -ne "$expected_episodes" ]]; then
  write_state invalid_base_report
  exit 4
fi
if [[ "$successes" -ge "$minimum_successes" ]]; then
  write_state base_floor_passed_no_stop
  exit 0
fi

# Freeze the coordinator before it can enter Anchor. The Base report is
# complete and immutable at this point; a seed manifest may be written just
# before or just after this signal, neither changes the failed floor.
kill -STOP "$paired_pid"
if [[ -n "$accelerator_pid" ]] && kill -0 "$accelerator_pid" 2>/dev/null; then
  kill -TERM "$accelerator_pid"
fi

python3 - "$final_root/base_floor_rejection.json" "$episodes" "$successes" \
  "$minimum_successes" <<'PY'
import json
import os
import sys
from pathlib import Path

destination, episodes, successes, minimum = sys.argv[1:]
episodes, successes, minimum = int(episodes), int(successes), int(minimum)
payload = {
    "schema": "zeva-robotwin-base-floor-rejection-v1",
    "base_complete": True,
    "total_episodes": episodes,
    "total_successes": successes,
    "success_rate": successes / episodes,
    "minimum_successes": minimum,
    "minimum_success_rate": minimum / episodes,
    "downstream_anchor_zeva_skipped": True,
    "reason": "Base failed the immutable normal-PI floor; downstream conditions cannot make this candidate deliverable",
}
path = Path(destination)
temporary = path.with_name(path.name + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY

write_state stopping_failed_paired_launcher
kill -TERM "$paired_pid"
kill -CONT "$paired_pid"
for _ in $(seq 1 20); do
  if ! kill -0 "$paired_pid" 2>/dev/null; then
    write_state stopped_after_complete_base_failure
    exit 0
  fi
  sleep 1
done
write_state failed_to_stop_paired_launcher
exit 5
