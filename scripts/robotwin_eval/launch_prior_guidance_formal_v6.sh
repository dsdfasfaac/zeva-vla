#!/usr/bin/env bash
set -euo pipefail

# Launch the fixed seed-1000 formal evaluation only after the two disjoint
# validation splits have selected the task-gated prior-only deployment adapter.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-prior-guidance-v6}
checkpoint=${CHECKPOINT:-$run_root/calibrated-checkpoint}
formal_root=${FORMAL_ROOT:-$run_root/formal-calibrated-prior05-v6}
formal_source=${FORMAL_SOURCE:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v5-multisplit/formal-calibrated-v5}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
model_host=${MODEL_HOST:-aigc29}
model_ip=${MODEL_IP:-172.16.80.163}
render_host=${RENDER_HOST:-aigc24}
render_runtime=${RENDER_RUNTIME:-/data1/dingxin/robotwin-formal-eval/RoboTwin}

test -s "$run_root/validation_summary.json"
test -s "$checkpoint/zeva_adapter.pth"
test -s "$formal_source/baseline/report.json"
test -s "$formal_source/seed_manifest.json"
ln -sfn \
  /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-safe-router-v3/zeva/001250/model.safetensors \
  "$checkpoint/model.safetensors"

python3 - "$run_root/validation_summary.json" "$checkpoint/zeva_adapter.pth" <<'PY'
import json
import sys

import torch

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if not summary["non_negative_each_split"] or summary["aggregate_gain"] < 2:
    raise SystemExit("v6 prior guidance did not replicate across validation splits")
checkpoint = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
expected = {"beat_block_hammer", "blocks_ranking_rgb"}
enabled = {
    task for task, scale in checkpoint["deployment_task_residual_scales"].items()
    if float(scale) > 0
}
if enabled != expected:
    raise SystemExit(f"unexpected enabled task set: {sorted(enabled)}")
if float(checkpoint["deployment_default_residual_scale"]) != 0.0:
    raise SystemExit("deployment default must be exact PI0.5 fallback")
if any(value.count_nonzero().item() for value in checkpoint["causal_action_projector"].values()):
    raise SystemExit("context branch must be exactly disabled")
if abs(torch.sigmoid(checkpoint["prior_gate_logit"]).item() - 0.5) > 1e-7:
    raise SystemExit("prior gate must equal BehaviorVLA guidance 0.5")
calibration = checkpoint["deployment_residual_calibration"]
if calibration.get("test_metrics_used") is not False:
    raise SystemExit("formal metrics must not be used for selection")
if calibration.get("seed_sets_pairwise_disjoint") is not True:
    raise SystemExit("validation and formal seeds must be disjoint")
PY

mkdir -p "$run_root/configs"
config=$run_root/configs/calibrated-prior05-v6.yml
python3 - "$config" "$checkpoint" <<'PY'
import sys
from pathlib import Path

destination, checkpoint = sys.argv[1:]
Path(destination).write_text(f"""policy_name: zeva_policy
handoff_root: /mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
foundation_checkpoint: /mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
goal_embedding_checkpoint: /mnt/100T/users/dingxin/VLA/runtime/pretrained_model-stage1-language-v1
stage2_checkpoint: {checkpoint}
zte_checkpoint: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/zte_best.pth
causal_bank: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/train_causal_bank.pt
retrieval_checkpoint: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1.5-task-retrieval/task_retrieval.pth
baseline_only: false
device: cuda
model_rng_seed: 20260907
""")
PY

env MODEL_HOST="$model_host" MODEL_IP="$model_ip" RENDER_HOST="$render_host" \
  RENDER_RUNTIME="$render_runtime" OUTPUT_ROOT="$formal_root" \
  BASELINE_CONFIG="$zeva_root/scripts/robotwin_eval/baseline_model_config.yml" \
  ZEVA_CONFIG="$config" ANCHOR_CONFIG="" TASK_MANIFEST="$task_manifest" \
  EPISODES=20 ABSOLUTE_START_SEED=1000 \
  FROZEN_SEED_MANIFEST="$formal_source/seed_manifest.json" \
  PRECOMPUTED_BASELINE_ROOT="$formal_source/baseline" \
  PRECOMPUTED_BASELINE_EXPECTED_SUCCESSES=114 \
  MIN_BASELINE_SUCCESS_RATE=0.57 \
  BASELINE_LABEL="immutable-normal-pi05-114-of-200" \
  ZEVA_LABEL="task-gated-prior05-v6" \
  bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

python3 "$zeva_root/scripts/robotwin_eval/audit_advantage10_formal.py" "$formal_root"
