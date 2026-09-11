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
  --task-manifest /tasks --base-pid 710436 --zeva-pid 751929 \
  --poll-seconds 1)
grep -q '^dry-run: no wait, write, SSH, selector, or formal evaluation will run$' <<<"$output"
grep -q '^formal_contract=episodes20,start_seed1000,' <<<"$output"
echo "wait_and_launch_robotwin_ztev2_formal: bash syntax and dry-run CLI passed"
