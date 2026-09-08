#!/usr/bin/env bash
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
foundation=${ROBOTWIN_FOUNDATION:-$handoff/checkpoint/pretrained_model-best-v1}
stage1_language=${ROBOTWIN_STAGE1_LANGUAGE:-$handoff/checkpoint/pretrained_model}
pair_root=${PAIR_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-best-v1-pair-v2-native-tf5}
task_subset=${ROBOTWIN_TASK_SUBSET:-$zeva_root/configs/robotwin_zeva_advantage10.json}
dataset_root=${ROBOTWIN_DATASET_ROOT:-/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data}

mkdir -p "$pair_root"

run_variant() {
  local variant=$1
  local output=$pair_root/$variant
  mkdir -p "$output"
  printf '%s\n' "starting $variant $(date --iso-8601=seconds)" | tee "$output/STARTED"
  local extra_args=()
  if [[ "$variant" == "zeva" ]]; then
    # Stage 1 was trained in the original PI task-language coordinate.  Keep
    # that auxiliary coordinate explicit while best-v1 remains the actual PI
    # vision-language/action foundation being optimized and deployed.
    extra_args+=(--goal-embedding-checkpoint "$stage1_language")
  fi
  bash "$zeva_root/scripts/train_robotwin_stage2_8gpu.sh" \
    --training-variant "$variant" \
    --foundation-checkpoint "$foundation" \
    --dataset-root "$dataset_root" \
    --task-subset "$task_subset" \
    --save-dir "$output" \
    --steps 1000 \
    --warmup-steps 250 \
    --save-freq 250 \
    --batch-size 16 \
    --gradient-accumulation-steps 2 \
    --num-workers 4 \
    --video-backend torchcodec \
    --baseline-preserve-interval 4 \
    --compile-model \
    --compile-mode default \
    --action-expert-learning-rate 1e-6 \
    --learning-rate 5e-5 \
    "${extra_args[@]}" \
    > "$output/train.log" 2>&1
  printf '%s\n' "completed $variant $(date --iso-8601=seconds)" | tee "$output/COMPLETE"
}

# A single eight-H100 node cannot hold both jobs concurrently.  Run them in a
# fixed order under the same outer launcher; Zeva starts only after the matched
# baseline completed successfully.
run_variant baseline
run_variant zeva
printf '%s\n' "completed pair $(date --iso-8601=seconds)" | tee "$pair_root/COMPLETE"
