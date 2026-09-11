#!/usr/bin/env bash
# Matched five-epoch learning curves; this never declares Stage1 acceptance.
set -euo pipefail
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
loss_reduction=${1:?Specify mean_coordinate_huber or vector_mse}
case "$loss_reduction" in
  mean_coordinate_huber|vector_mse) ;;
  *) echo "Unsupported loss control" >&2; exit 2 ;;
esac
run_dir=${ZEVA_TRAIN_RUN_DIR:?Set a fresh, explicit experiment directory}
: "${CUDA_VISIBLE_DEVICES:?Select four verified-free GPUs explicitly}"
: "${ZEVA_MAIN_PROCESS_PORT:?Select a distinct distributed port per simultaneous run}"
IFS=',' read -r -a selected_devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#selected_devices[@]}" -ne 4 ]]; then
  echo "This matched control requires exactly four GPUs" >&2
  exit 2
fi
if [[ -e "$run_dir/manifest.json" || -e "$run_dir/launcher.pid" ]]; then
  echo "Experiment already exists; inspect its live handle before resuming" >&2
  exit 2
fi
mkdir -p "$run_dir"
ZEVA_PROCESSES=4 nohup bash "$zeva_root/scripts/run_robotwin_zte_v2.sh" \
  --steps 4096 --batch-size 8 --num-workers 2 --seed 1000 \
  --warmup-steps 256 --save-freq 512 --eval-batches 0 --log-freq 16 \
  --action-prediction-context phase --prediction-loss-reduction "$loss_reduction" \
  --save-dir "$run_dir" > "$run_dir/train.log" 2>&1 < /dev/null &
training_pid=$!
printf '%s\n' "$training_pid" > "$run_dir/launcher.pid"
echo "Training launcher PID: $training_pid"
echo "Training log: $run_dir/train.log"
