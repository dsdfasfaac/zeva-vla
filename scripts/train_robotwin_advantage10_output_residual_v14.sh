#!/usr/bin/env bash
set -euo pipefail

# v14 direct output residual corrector.  Untouched PI0.5, Stage1 ZTE,
# task-language retrieval, causal bank, and Base action chunks stay frozen.
# ZeVA predicts a bounded normalized EEF16 delta from the exact H50 Base
# chunk plus task/phase/memory/Gaussian-prior features; only H15 is changed.
# H35 is copied from Base exactly.  The residual and gate heads are zero
# initialized, so the fresh adapter is an exact Base policy.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
foundation=${ROBOTWIN_FOUNDATION:-$handoff/checkpoint/pretrained_model-best-v1}
stage1_language=${ROBOTWIN_STAGE1_LANGUAGE:-/mnt/100T/users/dingxin/VLA/runtime/pretrained_model-stage1-language-v1}
dataset_root=${ROBOTWIN_DATASET_ROOT:-/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data}
task_subset=${ROBOTWIN_TASK_SUBSET:-$zeva_root/configs/robotwin_zeva_advantage10.json}
stage1_root=${ROBOTWIN_STAGE1_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1}
run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-output-residual-v14}
output=$run_root/zeva
base_action_cache=${BASE_ACTION_CACHE:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-output-correction-v13/base_action_cache.pt}
foundation_sha256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe

for required in \
  "$foundation/model.safetensors" \
  "$stage1_language/model.safetensors" \
  "$dataset_root/adapter.json" \
  "$task_subset" \
  "$stage1_root/stage1-zte/zte_best.pth" \
  "$stage1_root/stage1-zte/train_causal_bank.pt" \
  "$stage1_root/stage1-zte/live_queries_h15.pt" \
  "$stage1_root/stage1.5-task-retrieval/task_retrieval.pth"; do
  test -s "$required"
done
test -s "$base_action_cache"
test "$(sha256sum "$foundation/model.safetensors" | awk '{print $1}')" = "$foundation_sha256"

if [[ -e "$output/manifest.json" ]]; then
  echo "refusing to overwrite existing v14 run: $output" >&2
  exit 3
fi

steps=${STEPS:-1500}
save_freq=${SAVE_FREQ:-250}
eval_batches=${EVAL_BATCHES:-64}
batch_size=16
gradient_accumulation_steps=2
compile_args=(--compile-model --compile-mode default)
if [[ "${V14_SMOKE:-0}" == 1 ]]; then
  steps=1
  save_freq=1
  eval_batches=1
  batch_size=32
  gradient_accumulation_steps=1
  compile_args=(--no-compile-model)
fi

mkdir -p "$output"
printf '%s\n' "starting output-residual v14 $(date --iso-8601=seconds)" | tee "$output/STARTED"

bash "$zeva_root/scripts/train_robotwin_stage2_8gpu.sh" \
  --training-variant output_residual \
  --foundation-checkpoint "$foundation" \
  --goal-embedding-checkpoint "$stage1_language" \
  --dataset-root "$dataset_root" \
  --task-subset "$task_subset" \
  --zte-checkpoint "$stage1_root/stage1-zte/zte_best.pth" \
  --causal-bank "$stage1_root/stage1-zte/train_causal_bank.pt" \
  --live-queries "$stage1_root/stage1-zte/live_queries_h15.pt" \
  --task-retrieval "$stage1_root/stage1.5-task-retrieval/task_retrieval.pth" \
  --base-action-cache "$base_action_cache" \
  --save-dir "$output" \
  --steps "$steps" \
  --warmup-steps "${WARMUP_STEPS:-100}" \
  --save-freq "$save_freq" \
  --batch-size "$batch_size" \
  --gradient-accumulation-steps "$gradient_accumulation_steps" \
  --num-workers 4 \
  --video-backend torchcodec \
  --prior-loss-weight "${PRIOR_LOSS_WEIGHT:-0.01}" \
  --preserve-loss-weight "${PRESERVE_LOSS_WEIGHT:-8.0}" \
  --paired-improvement-margin "${PAIRED_IMPROVEMENT_MARGIN:-0.0}" \
  --gate-regularization-weight "${GATE_REGULARIZATION_WEIGHT:-0.0001}" \
  --residual-bound "${RESIDUAL_BOUND:-0.25}" \
  --residual-regression-weight "${RESIDUAL_REGRESSION_WEIGHT:-0.25}" \
  --residual-trust-region-weight "${RESIDUAL_TRUST_REGION_WEIGHT:-0.01}" \
  --residual-trust-region-radius "${RESIDUAL_TRUST_REGION_RADIUS:-0.10}" \
  --prior-residual-dropout-probability 0 \
  --prior-injection-horizon 15 \
  --eval-batches "$eval_batches" \
  --learning-rate "${ZEVA_LR:-5e-5}" \
  --action-expert-learning-rate 5e-6 \
  "${compile_args[@]}" \
  >> "$output/train.log" 2>&1

printf '%s\n' "completed output-residual v14 $(date --iso-8601=seconds)" | tee "$output/COMPLETE"
