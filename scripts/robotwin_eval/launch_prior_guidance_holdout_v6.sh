#!/usr/bin/env bash
set -euo pipefail

# Run one additional, disjoint closed-loop validation split for the already
# frozen v6 deployment candidate.  This script never reads the seed-1000
# formal outcomes and never changes the adapter.  Its purpose is to measure
# whether the validation-only gain replicates under a fresh expert-valid seed
# stream before any later model revision is considered.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-prior-guidance-v6}
output_root=${OUTPUT_ROOT:-$run_root/context0-prior05/split-c}
candidate_config=${CANDIDATE_CONFIG:-$run_root/configs/calibrated-prior05-v6.yml}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
baseline_config=${BASELINE_CONFIG:-$zeva_root/scripts/robotwin_eval/baseline_model_config.yml}

model_host=${MODEL_HOST:-aigc32}
model_ip=${MODEL_IP:-172.16.80.166}
render_host=${RENDER_HOST:-aigc15}
render_runtime=${RENDER_RUNTIME:-/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin}
episodes=${EPISODES:-8}
absolute_start_seed=${ABSOLUTE_START_SEED:-4000}

test -s "$candidate_config"
test -s "$task_manifest"
test -s "$baseline_config"
mkdir -p "$output_root"

cat > "$output_root/holdout_plan.json" <<EOF
{
  "schema": "zeva-robotwin-frozen-candidate-holdout-v1",
  "candidate": "task-gated-prior05-v6",
  "candidate_config": "$candidate_config",
  "adapter_is_frozen": true,
  "formal_test_metrics_used": false,
  "episodes_per_task": $episodes,
  "absolute_start_seed": $absolute_start_seed,
  "model_host": "$model_host",
  "render_host": "$render_host",
  "purpose": "independent replication of the already-frozen v6 validation result"
}
EOF

env MODEL_HOST="$model_host" MODEL_IP="$model_ip" \
  RENDER_HOST="$render_host" RENDER_RUNTIME="$render_runtime" \
  OUTPUT_ROOT="$output_root" BASELINE_CONFIG="$baseline_config" \
  ZEVA_CONFIG="$candidate_config" ANCHOR_CONFIG="" \
  TASK_MANIFEST="$task_manifest" EPISODES="$episodes" \
  ABSOLUTE_START_SEED="$absolute_start_seed" \
  BASELINE_LABEL="closed-loop-validation-split-c-pi05" \
  ZEVA_LABEL="task-gated-prior05-v6-validation-split-c" \
  bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

