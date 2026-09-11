#!/usr/bin/env bash
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-pilot-20260911d
if [[ -e "$run_dir/manifest.json" || -e "$run_dir/launcher.pid" ]]; then
  echo "Pilot already exists; inspect its process/checkpoint before resuming." >&2
  exit 2
fi
mkdir -p "$run_dir"
# A fixed, limited learning-curve pilot, not a complete Stage1 claim. Four
# GPUs leave GPU2 free for independent real-PI injection validation.
CUDA_VISIBLE_DEVICES=0,1,3,4 ZEVA_PROCESSES=4 \
  nohup bash "$zeva_root/scripts/run_robotwin_zte_v2.sh" \
    --steps 256 --batch-size 8 --num-workers 2 \
    --warmup-steps 32 --save-freq 128 --eval-batches 0 --log-freq 8 \
    --save-dir "$run_dir" \
    > "$run_dir/train.log" 2>&1 < /dev/null &
pilot_pid=$!
printf '%s\n' "$pilot_pid" > "$run_dir/launcher.pid"
echo "Pilot launcher PID: $pilot_pid"
echo "Pilot log: $run_dir/train.log"
