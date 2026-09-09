#!/usr/bin/env bash
set -euo pipefail

# Retrain the isolated Gaussian action-prior residual at the exact gate used in
# deployment.  PI0.5, ZTE/Mamba, the causal bank, task retrieval, and the
# context-residual branch remain frozen.  Selection remains validation-only.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
foundation=${ROBOTWIN_FOUNDATION:-/data1/dingxin/zeva-checkpoint-cache-candidate-v1/pretrained_model-best-v1}
stage1_language=${ROBOTWIN_STAGE1_LANGUAGE:-/data1/dingxin/zeva-checkpoint-cache-candidate-v1/pretrained_model-stage1-language-v1}
dataset_root=${ROBOTWIN_DATASET_ROOT:-/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data}
task_subset=${ROBOTWIN_TASK_SUBSET:-$zeva_root/configs/robotwin_zeva_advantage10.json}
stage1_root=${ROBOTWIN_STAGE1_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1}
output=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-prior-only-v7}/zeva

mkdir -p "$output"
printf '%s\n' "starting fixed-gate prior-only v7 $(date --iso-8601=seconds)" | tee -a "$output/STARTED"

bash "$zeva_root/scripts/train_robotwin_stage2_8gpu.sh" \
  --training-variant prior_adapter \
  --foundation-checkpoint "$foundation" \
  --goal-embedding-checkpoint "$stage1_language" \
  --dataset-root "$dataset_root" \
  --task-subset "$task_subset" \
  --zte-checkpoint "$stage1_root/stage1-zte/zte_best.pth" \
  --causal-bank "$stage1_root/stage1-zte/train_causal_bank.pt" \
  --live-queries "$stage1_root/stage1-zte/live_queries_h15.pt" \
  --task-retrieval "$stage1_root/stage1.5-task-retrieval/task_retrieval.pth" \
  --save-dir "$output" \
  --steps "${STEPS:-2000}" \
  --warmup-steps "${WARMUP_STEPS:-250}" \
  --save-freq "${SAVE_FREQ:-250}" \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --num-workers 4 \
  --video-backend torchcodec \
  --baseline-preserve-interval "${BASELINE_PRESERVE_INTERVAL:-2}" \
  --preserve-loss-weight "${PRESERVE_LOSS_WEIGHT:-1.0}" \
  --gate-regularization-weight 0 \
  --prior-loss-weight "${PRIOR_LOSS_WEIGHT:-0.01}" \
  --prior-residual-dropout-probability "${PRIOR_DROPOUT:-0.4}" \
  --eval-batches "${EVAL_BATCHES:-64}" \
  --compile-model \
  --compile-mode default \
  --action-expert-learning-rate 5e-8 \
  --learning-rate "${ZEVA_LR:-5e-5}" \
  >> "$output/train.log" 2>&1

printf '%s\n' "completed fixed-gate prior-only v7 $(date --iso-8601=seconds)" | tee -a "$output/COMPLETE"

