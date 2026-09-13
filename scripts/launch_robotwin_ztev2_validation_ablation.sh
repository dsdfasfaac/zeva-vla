#!/usr/bin/env bash
set -euo pipefail
set -o noclobber

# One-shot validation launch; no scheduling, training, or checkpoint mutation.
if [[ $# != 2 ]]; then
  echo "Usage: bash $0 NEW_OUTPUT_DIRECTORY VERIFIED_DATASET_ROOT" >&2
  exit 2
fi
zeva_output=$1
zeva_data=$2
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
zeva_pair=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-ztev2-schedulerfix-pair-20260911
# Keep four concurrent model loads from oversubscribing every host CPU core.
# The same limits apply to all ablations and their fresh same-host control.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
[[ ! -e "$zeva_output" && -f "$zeva_data/adapter.json" ]] || exit 2
for zeva_gpu in 0 1 2 3; do
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v target="$zeva_gpu" '
      $1+0 == target { found=1; if ($2+0 > 1024 || $3+0 > 5) exit 1 }
      END { if (!found) exit 1 }' || { echo "GPU $zeva_gpu is not idle." >&2; exit 1; }
done
mkdir "$zeva_output"
cp "$zeva_root/configs/robotwin_ztev2_validation_ablation_20260913.json" "$zeva_output/plan.json"
for zeva_spec in 'context_only 0 1 0' 'prior_only 1 0 1' 'prior_strength_test 2 1 50' 'same_host_control 3 1 1'; do
  read -r zeva_name zeva_gpu zeva_context zeva_prior <<< "$zeva_spec"
  CUDA_VISIBLE_DEVICES="$zeva_gpu" nohup bash "$zeva_root/scripts/run_robotwin_stage2_diagnostics.sh" \
    --checkpoint "$zeva_pair/zeva/005000" --manifest "$zeva_pair/zeva/manifest.json" \
    --fixed-teacher-checkpoint "$zeva_pair/baseline/004500" --dataset-root "$zeva_data" \
    --batch-size 8 --eval-batches 0 --seed 1000 \
    --context-gate-scale "$zeva_context" --prior-gate-scale "$zeva_prior" \
    --output "$zeva_output/$zeva_name.json" > "$zeva_output/$zeva_name.log" 2>&1 < /dev/null &
  zeva_pid=$!
  printf '%s\n' "$zeva_pid" > "$zeva_output/$zeva_name.pid"
  printf '%s GPU=%s PID=%s\n' "$zeva_name" "$zeva_gpu" "$zeva_pid"
done
