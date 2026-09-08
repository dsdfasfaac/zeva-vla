#!/usr/bin/env bash
set -euo pipefail

# Conservative ten-task adaptation for an already RoboTwin-trained PI0.5.
# Run baseline and ZeVA on separate eight-H100 nodes.  Their action experts use
# the same schedule; only ZeVA owns the additional high-LR residual modules.

variant=${1:?usage: train_robotwin_advantage10_conservative_variant.sh baseline|zeva}
if [[ "$variant" != baseline && "$variant" != zeva ]]; then
  echo "variant must be baseline or zeva" >&2
  exit 2
fi

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
foundation=${ROBOTWIN_FOUNDATION:-/data1/dingxin/zeva-checkpoint-cache-candidate-v1/pretrained_model-best-v1}
stage1_language=${ROBOTWIN_STAGE1_LANGUAGE:-/data1/dingxin/zeva-checkpoint-cache-candidate-v1/pretrained_model-stage1-language-v1}
dataset_root=${ROBOTWIN_DATASET_ROOT:-/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data}
task_subset=${ROBOTWIN_TASK_SUBSET:-$zeva_root/configs/robotwin_zeva_advantage10.json}
stage1_root=${ROBOTWIN_STAGE1_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1}
zte_checkpoint=${ROBOTWIN_ZTE_CHECKPOINT:-$stage1_root/stage1-zte/zte_best.pth}
causal_bank=${ROBOTWIN_CAUSAL_BANK:-$stage1_root/stage1-zte/train_causal_bank.pt}
live_queries=${ROBOTWIN_LIVE_QUERIES:-$stage1_root/stage1-zte/live_queries_h15.pt}
task_retrieval=${ROBOTWIN_TASK_RETRIEVAL:-$stage1_root/stage1.5-task-retrieval/task_retrieval.pth}
action_expert_learning_rate=${ACTION_EXPERT_LR:-1e-7}
zeva_learning_rate=${ZEVA_LR:-5e-5}
steps=${STEPS:-250}
warmup_steps=${WARMUP_STEPS:-250}
save_freq=${SAVE_FREQ:-125}
run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-conservative-ae1e-7-v1}
output=$run_root/$variant

mkdir -p "$output"
printf '%s\n' "starting $variant $(date --iso-8601=seconds)" | tee "$output/STARTED"
extra_args=()
if [[ "$variant" == zeva ]]; then
  extra_args+=(--goal-embedding-checkpoint "$stage1_language")
fi

bash "$zeva_root/scripts/train_robotwin_stage2_8gpu.sh" \
  --training-variant "$variant" \
  --foundation-checkpoint "$foundation" \
  --dataset-root "$dataset_root" \
  --task-subset "$task_subset" \
  --zte-checkpoint "$zte_checkpoint" \
  --causal-bank "$causal_bank" \
  --live-queries "$live_queries" \
  --task-retrieval "$task_retrieval" \
  --save-dir "$output" \
  --steps "$steps" \
  --warmup-steps "$warmup_steps" \
  --save-freq "$save_freq" \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --num-workers 4 \
  --video-backend torchcodec \
  --baseline-preserve-interval 4 \
  --compile-model \
  --compile-mode default \
  --action-expert-learning-rate "$action_expert_learning_rate" \
  --learning-rate "$zeva_learning_rate" \
  "${extra_args[@]}" \
  > "$output/train.log" 2>&1

printf '%s\n' "completed $variant $(date --iso-8601=seconds)" | tee "$output/COMPLETE"
