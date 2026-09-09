#!/usr/bin/env bash
set -euo pipefail

# Matched ten-task Stage 2 training from the released RoboTwin best-v1 PI0.5.
# Both variants tune the action expert with the PaliGemma vision-language
# backbone frozen.  The ZeVA variant additionally trains the task/phase-routed
# context and Gaussian action-prior residuals; Stage 1 and retrieval stay
# frozen.  Run the two variants on separate eight-H100 hosts.

variant=${1:?usage: train_robotwin_advantage10_action_expert_v8.sh baseline|zeva}
if [[ "$variant" != baseline && "$variant" != zeva ]]; then
  echo "variant must be baseline or zeva" >&2
  exit 2
fi

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
foundation=${ROBOTWIN_FOUNDATION:-/data1/dingxin/zeva-checkpoint-cache-candidate-v1/pretrained_model-best-v1}
foundation_sha256=${ROBOTWIN_FOUNDATION_SHA256:-7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe}
stage1_language=${ROBOTWIN_STAGE1_LANGUAGE:-/data1/dingxin/zeva-checkpoint-cache-candidate-v1/pretrained_model-stage1-language-v1}
dataset_root=${ROBOTWIN_DATASET_ROOT:-/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data}
task_subset=${ROBOTWIN_TASK_SUBSET:-$zeva_root/configs/robotwin_zeva_advantage10.json}
stage1_root=${ROBOTWIN_STAGE1_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1}
run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-action-expert-v8}
output=$run_root/$variant
resume_checkpoint=${RESUME_CHECKPOINT:-}

for required in \
  "$foundation/model.safetensors" \
  "$dataset_root/adapter.json" \
  "$task_subset" \
  "$stage1_root/stage1-zte/zte_best.pth" \
  "$stage1_root/stage1-zte/train_causal_bank.pt" \
  "$stage1_root/stage1-zte/live_queries_h15.pt" \
  "$stage1_root/stage1.5-task-retrieval/task_retrieval.pth"; do
  test -s "$required"
done
actual_foundation_sha256=$(sha256sum "$foundation/model.safetensors" | awk '{print $1}')
if [[ "$actual_foundation_sha256" != "$foundation_sha256" ]]; then
  echo "unexpected best-v1 foundation hash: $actual_foundation_sha256" >&2
  exit 4
fi
if [[ "$variant" == zeva ]]; then
  test -s "$stage1_language/model.safetensors"
fi
if [[ -e "$output/COMPLETE" ]]; then
  echo "v8 run is already complete: $output" >&2
  exit 0
fi
if [[ -e "$output/manifest.json" && -z "$resume_checkpoint" ]]; then
  echo "refusing to overwrite existing v8 run: $output" >&2
  exit 3
fi
if [[ -n "$resume_checkpoint" ]]; then
  test -s "$resume_checkpoint/training_state.pt"
  if [[ "$(realpath "$(dirname "$resume_checkpoint")")" != "$(realpath "$output")" ]]; then
    echo "resume checkpoint must belong to $output: $resume_checkpoint" >&2
    exit 5
  fi
fi

mkdir -p "$output"
printf '%s\n' "starting action-expert v8 $variant $(date --iso-8601=seconds) resume=${resume_checkpoint:-none}" \
  | tee -a "$output/STARTED"

extra_args=()
if [[ "$variant" == zeva ]]; then
  # Stage 1 was learned in this frozen task-language coordinate.  It is an
  # auxiliary lookup coordinate, not a second trainable language backbone.
  extra_args+=(--goal-embedding-checkpoint "$stage1_language")
fi
if [[ -n "$resume_checkpoint" ]]; then
  extra_args+=(--resume-checkpoint "$resume_checkpoint")
fi

bash "$zeva_root/scripts/train_robotwin_stage2_8gpu.sh" \
  --training-variant "$variant" \
  --foundation-checkpoint "$foundation" \
  --dataset-root "$dataset_root" \
  --task-subset "$task_subset" \
  --zte-checkpoint "$stage1_root/stage1-zte/zte_best.pth" \
  --causal-bank "$stage1_root/stage1-zte/train_causal_bank.pt" \
  --live-queries "$stage1_root/stage1-zte/live_queries_h15.pt" \
  --task-retrieval "$stage1_root/stage1.5-task-retrieval/task_retrieval.pth" \
  --save-dir "$output" \
  --steps "${STEPS:-5000}" \
  --warmup-steps "${WARMUP_STEPS:-250}" \
  --save-freq "${SAVE_FREQ:-500}" \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --num-workers 4 \
  --video-backend torchcodec \
  --baseline-preserve-interval "${BASELINE_PRESERVE_INTERVAL:-2}" \
  --preserve-loss-weight "${PRESERVE_LOSS_WEIGHT:-4.0}" \
  --gate-regularization-weight "${GATE_REGULARIZATION_WEIGHT:-0.001}" \
  --prior-loss-weight "${PRIOR_LOSS_WEIGHT:-0.01}" \
  --prior-residual-dropout-probability "${PRIOR_DROPOUT:-0.4}" \
  --eval-batches "${EVAL_BATCHES:-64}" \
  --compile-model \
  --compile-mode default \
  --action-expert-learning-rate "${ACTION_EXPERT_LR:-5e-6}" \
  --learning-rate "${ZEVA_LR:-5e-5}" \
  "${extra_args[@]}" \
  >> "$output/train.log" 2>&1

printf '%s\n' "completed action-expert v8 $variant $(date --iso-8601=seconds)" \
  | tee "$output/COMPLETE"
