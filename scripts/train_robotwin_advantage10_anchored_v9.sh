#!/usr/bin/env bash
set -euo pipefail

# v9 starts both candidates from the held-out-selected action-expert Base and
# learns a fresh dual ZeVA residual.  The joint candidate keeps an immutable
# copy of that Base action path as its paired teacher; the adapter candidate
# freezes the same action path and therefore uses its own residual-off forward
# as an exact Base teacher.  PaliGemma, Stage 1 ZTE/Mamba, causal bank, live H15
# queries, and task-language retrieval remain frozen in both cases.

mode=${1:?usage: train_robotwin_advantage10_anchored_v9.sh joint|adapter}
case "$mode" in
  joint) training_variant=zeva ;;
  adapter) training_variant=adapter ;;
  *) echo "mode must be joint or adapter" >&2; exit 2 ;;
esac

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
foundation=${ROBOTWIN_FOUNDATION:-/data1/dingxin/zeva-checkpoint-cache-candidate-v1/pretrained_model-best-v1}
foundation_sha256=${ROBOTWIN_FOUNDATION_SHA256:-7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe}
stage1_language=${ROBOTWIN_STAGE1_LANGUAGE:-/data1/dingxin/zeva-checkpoint-cache-candidate-v1/pretrained_model-stage1-language-v1}
dataset_root=${ROBOTWIN_DATASET_ROOT:-/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data}
task_subset=${ROBOTWIN_TASK_SUBSET:-$zeva_root/configs/robotwin_zeva_advantage10.json}
stage1_root=${ROBOTWIN_STAGE1_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1}
base_checkpoint=${ROBOTWIN_TRAINED_BASE:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-action-expert-v8/baseline/003000}
run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-anchored-v9}
output=$run_root/$mode
resume_checkpoint=${RESUME_CHECKPOINT:-}

for required in \
  "$foundation/model.safetensors" \
  "$stage1_language/model.safetensors" \
  "$base_checkpoint/model.safetensors" \
  "$base_checkpoint/training_state.pt" \
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
if [[ -e "$output/COMPLETE" ]]; then
  echo "v9 run is already complete: $output" >&2
  exit 0
fi
if [[ -e "$output/manifest.json" && -z "$resume_checkpoint" ]]; then
  echo "refusing to overwrite existing v9 run: $output" >&2
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
printf '%s\n' "starting anchored v9 $mode $(date --iso-8601=seconds) resume=${resume_checkpoint:-none}" \
  | tee -a "$output/STARTED"

extra_args=()
if [[ "$mode" == joint ]]; then
  extra_args+=(--anchor-stage2-checkpoint "$base_checkpoint")
fi
if [[ -n "$resume_checkpoint" ]]; then
  extra_args+=(--resume-checkpoint "$resume_checkpoint")
fi
if [[ "${SAVE_CHECKPOINTS:-1}" == 0 ]]; then
  extra_args+=(--no-save-checkpoints)
fi

bash "$zeva_root/scripts/train_robotwin_stage2_8gpu.sh" \
  --training-variant "$training_variant" \
  --foundation-checkpoint "$foundation" \
  --initial-stage2-checkpoint "$base_checkpoint" \
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
  --preserve-loss-weight "${PRESERVE_LOSS_WEIGHT:-8.0}" \
  --paired-improvement-margin "${PAIRED_IMPROVEMENT_MARGIN:-0.00005}" \
  --gate-regularization-weight "${GATE_REGULARIZATION_WEIGHT:-0.00001}" \
  --initial-residual-gate-probability "${INITIAL_RESIDUAL_GATE_PROBABILITY:-0.25}" \
  --prior-loss-weight "${PRIOR_LOSS_WEIGHT:-0.01}" \
  --prior-residual-dropout-probability "${PRIOR_DROPOUT:-0.4}" \
  --eval-batches "${EVAL_BATCHES:-64}" \
  --compile-model \
  --compile-mode default \
  --action-expert-learning-rate "${ACTION_EXPERT_LR:-5e-7}" \
  --learning-rate "${ZEVA_LR:-5e-5}" \
  "${extra_args[@]}" \
  >> "$output/train.log" 2>&1

printf '%s\n' "completed anchored v9 $mode $(date --iso-8601=seconds)" \
  | tee "$output/COMPLETE"
