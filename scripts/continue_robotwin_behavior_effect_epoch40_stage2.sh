#!/usr/bin/env bash
# Await the already-running epoch40 export, then fail closed through audit/training.
set -euo pipefail
[[ $(hostname -s) == aigc28 ]] || { echo 'Continuation is restricted to aigc28' >&2; exit 2; }
export_pid=${1:?Usage: continue_robotwin_behavior_effect_epoch40_stage2.sh EXPORT_PID}
[[ "$export_pid" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid export PID' >&2; exit 2; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-behavior-effect-20260918
artifact=$run/cte-epoch040-exploratory-artifacts.pth
launcher=$root/scripts/run_robotwin_behavior_effect_epoch40_stage2.sh

for ((attempt=0; attempt<720; attempt++)); do
  if [[ -s "$artifact" ]]; then
    break
  fi
  exporter_command=$(ps -p "$export_pid" -o args= || true)
  if [[ "$exporter_command" != *"export_robotwin_behavior_effect.py"* || "$exporter_command" != *"$artifact"* ]]; then
    echo 'Exporter exited before its atomic artifact appeared; do not start Stage2' >&2
    exit 2
  fi
  sleep 30
done
[[ -s "$artifact" ]] || { echo 'Timed out waiting for epoch40 artifact' >&2; exit 2; }
echo "epoch40 artifact ready: $artifact"
/bin/bash "$launcher" preflight
echo 'Independent epoch40 preflight passed; checking four idle GPUs and starting Stage2'
/bin/bash "$launcher" stage2
