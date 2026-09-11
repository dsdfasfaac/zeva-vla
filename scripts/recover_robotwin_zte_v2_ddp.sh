#!/usr/bin/env bash
# Exact-RNG Stage1 v2 recovery after an epoch-end all-padding DDP reduction.
#
# The source run is never overwritten.  The resumed checkpoint, schedule,
# world size, and objective arguments are fixed to the failed 4096-step run;
# only the output directory and distributed port are changed.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_run=${ZEVA_RECOVERY_SOURCE_RUN:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-phase-vector-mse-4096-20260911h}
resume_checkpoint=${ZEVA_RECOVERY_CHECKPOINT:-$source_run/zte_v2_step_000512.pth}
run_dir=${ZEVA_RECOVERY_RUN_DIR:-${source_run}-ddp-recovery}
log_path=$run_dir/train.log

test -f "$resume_checkpoint"
if [[ -e "$run_dir/manifest.json" || -e "$run_dir/launcher.pid" ]]; then
  echo "Recovery output already exists; refusing to overwrite: $run_dir" >&2
  exit 2
fi
mkdir -p "$run_dir"

devices=${CUDA_VISIBLE_DEVICES:-0,1,3,4}
processes=${ZEVA_PROCESSES:-4}
port=${ZEVA_MAIN_PROCESS_PORT:-29511}
if [[ "$processes" -ne 4 ]]; then
  echo "This recovery is fixed to world size 4, got $processes" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="$devices" ZEVA_PROCESSES=4 ZEVA_MAIN_PROCESS_PORT="$port" \
  nohup bash "$zeva_root/scripts/run_robotwin_zte_v2.sh" \
    --steps 4096 --batch-size 8 --num-workers 2 --seed 1000 \
    --warmup-steps 256 --save-freq 512 --eval-batches 0 --log-freq 16 \
    --action-prediction-context phase --prediction-loss-reduction vector_mse \
    --resume-checkpoint "$resume_checkpoint" --save-dir "$run_dir" \
    > "$log_path" 2>&1 < /dev/null &
recovery_pid=$!
printf '%s\n' "$recovery_pid" > "$run_dir/launcher.pid"
printf 'Recovery launcher PID: %s\nRecovery log: %s\nRecovery output: %s\n' \
  "$recovery_pid" "$log_path" "$run_dir"
