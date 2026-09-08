#!/usr/bin/env bash
set -euo pipefail

eval_parent=${EVAL_PARENT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5}
main_root=${MAIN_ROOT:-$eval_parent/formal-b1000-z250-seeded-v1}
output=${OUTPUT_ROOT:-$eval_parent/formal-conservative-b125-z250-seeded-v1}
checkpoint_cache=${CHECKPOINT_CACHE:-/data1/dingxin/zeva-checkpoint-cache-conservative-v1}
zeva_root=/mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA

while [[ ! -f "$checkpoint_cache/COMPLETE" ]]; do
  sleep 20
done
while ! grep -q '"state":"complete"' "$main_root/state.json" 2>/dev/null; do
  sleep 30
done

exec env \
  MODEL_HOST=${MODEL_HOST:-aigc28} \
  RENDER_HOST=${RENDER_HOST:-aigc24} \
  MODEL_IP=${MODEL_IP:-172.16.80.162} \
  RENDER_RUNTIME=${RENDER_RUNTIME:-/data1/dingxin/robotwin-formal-eval/RoboTwin} \
  RENDER_VULKAN_ICD=${RENDER_VULKAN_ICD:-/usr/share/vulkan/icd.d/nvidia_icd.json} \
  SLOTS=${SLOTS:-8} \
  BASE_PORT=${BASE_PORT:-19400} \
  EPISODES=20 \
  OUTPUT_ROOT="$output" \
  BASELINE_CONFIG="$zeva_root/configs/robotwin_eval_advantage10_conservative_base125.yml" \
  ANCHOR_CONFIG="$zeva_root/configs/robotwin_eval_advantage10_conservative_anchor.yml" \
  ZEVA_CONFIG="$zeva_root/configs/robotwin_eval_advantage10_conservative_zeva250.yml" \
  TASK_MANIFEST="$zeva_root/configs/robotwin_zeva_advantage10.json" \
  FROZEN_SEED_MANIFEST="$main_root/seed_manifest.json" \
  BASELINE_LABEL=advantage10-conservative-baseline-step125 \
  ANCHOR_LABEL=pretrained-model-best-v1-anchor \
  ZEVA_LABEL=advantage10-conservative-zeva-step250 \
  MODEL_RNG_SEED=20260907 \
  MODEL_SEED_POLICY=continuous \
  bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"
