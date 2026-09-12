#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
script="$script_dir/wait_and_launch_robotwin_ztev2_formal.sh"
bash -n "$script"
output=$("$script" --dry-run \
  --zeva-root /repo --train-root /train --output-root /out \
  --handoff-root /handoff --foundation-checkpoint /foundation \
  --goal-embedding-checkpoint /goal --zte-checkpoint /zte \
  --causal-bank /bank --retrieval-checkpoint /retrieval \
  --task-manifest /tasks --base-pid 0 --zeva-pid 751929 \
  --poll-seconds 1)
grep -q '^dry-run: no wait, write, SSH, selector, or formal evaluation will run$' <<<"$output"
grep -q '^formal_contract=episodes20,start_seed1000,' <<<"$output"
grep -q '^zeva_root=/repo$' <<<"$output"
grep -q 'base_pid=0' <<<"$output"

if negative_output=$("$script" --dry-run \
  --zeva-root /repo --train-root /train --output-root /out \
  --handoff-root /handoff --foundation-checkpoint /foundation \
  --goal-embedding-checkpoint /goal --zte-checkpoint /zte \
  --causal-bank /bank --retrieval-checkpoint /retrieval \
  --task-manifest /tasks --base-pid -1 --zeva-pid 751929 \
  --poll-seconds 1 2>&1); then
  echo "negative PID unexpectedly accepted" >&2
  exit 1
fi
grep -q -- '--base-pid must be 0 or a positive PID' <<<"$negative_output"
echo "wait_and_launch_robotwin_ztev2_formal: bash syntax and dry-run CLI passed"
