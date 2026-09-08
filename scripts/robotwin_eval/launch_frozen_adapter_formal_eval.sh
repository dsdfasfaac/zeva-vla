#!/usr/bin/env bash
set -euo pipefail

# Select from validation5 only, cache the immutable artifact on the model host,
# then run the single final 10-task x 20-episode Base-versus-ZeVA comparison.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
train_root=${TRAIN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-safe-router-v3/zeva}
eval_parent=${EVAL_PARENT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v3}
output=${OUTPUT_ROOT:-$eval_parent/formal-selected-validation5-v3}
model_host=${MODEL_HOST:-aigc29}
render_host=${RENDER_HOST:-aigc24}
foundation=${FOUNDATION_CHECKPOINT:-/data1/dingxin/zeva-checkpoint-cache/pretrained_model-best-v1}
goal_embedding=${GOAL_EMBEDDING_CHECKPOINT:-/data1/dingxin/zeva-checkpoint-cache/pretrained_model-stage1-language-v1}
stage1_root=${STAGE1_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1}
historical_root=${HISTORICAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5/formal-b1000-z250-seeded-v1}
selection=$eval_parent/validation5_selection.json
adapter_cache_root=${ADAPTER_CACHE_ROOT:-/data1/dingxin/zeva-checkpoint-cache-safe-router-v3}

if [[ ! -f "$train_root/COMPLETE" ]]; then
  echo "training is not complete: $train_root/COMPLETE is missing" >&2
  exit 3
fi
mkdir -p "$eval_parent/configs"

selected=$(python3 - "$train_root" "$selection" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

import torch

root, destination = map(Path, sys.argv[1:])
rows = []
for checkpoint_dir in sorted(root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]")):
    step = int(checkpoint_dir.name)
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.is_file():
        raise RuntimeError(f"missing preregistered checkpoint {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    validation = state["validation"]
    row = {"step": step, "checkpoint": str(state_path.parent), **validation}
    row["paired_flow_improvement"] = float(validation["baseline"]) - float(validation["flow"])
    task_rows = validation.get("per_task_paired", {})
    expected_tasks = set(state["manifest"]["task_scope"]["task_names"])
    row["complete_task_coverage"] = set(task_rows) == expected_tasks
    row["all_tasks_non_regressing"] = row["complete_task_coverage"] and all(
        float(task["paired_improvement"]) >= 0.0 for task in task_rows.values()
    )
    row["qualified"] = (
        math.isfinite(float(validation["flow"]))
        and float(validation["retrieval_accuracy"]) >= 0.95
        and row["paired_flow_improvement"] > 0.0
        and float(validation["paired_win_fraction"]) >= 0.5
        and row["all_tasks_non_regressing"]
    )
    rows.append(row)
qualified = [row for row in rows if row["qualified"]]
if not qualified:
    raise RuntimeError("no checkpoint passes retrieval>=0.95 and enhanced-flow<=frozen-PI")
# Absolute flow at different checkpoints uses different training RNG. The
# matched baseline and enhanced branch within a checkpoint share exact noise,
# so paired improvement is the comparable non-regression signal.
chosen = min(
    qualified,
    key=lambda row: (
        -row["paired_win_fraction"],
        -row["minimum_task_paired_improvement"],
        row["paired_degradation"],
        -row["paired_flow_improvement"],
        row["prior"],
        row["step"],
    ),
)
payload = {
    "schema": "zeva-frozen-pi-validation5-selection-v2",
    "selection_rule": "full ten-task validation coverage, retrieval>=0.95, every task paired improvement>=0, per-example win fraction>=0.5, mean paired improvement>0; max win fraction, max worst-task improvement, min degradation, max mean improvement, min prior, min step",
    "candidates": rows,
    "selected": chosen,
    "test_metrics_used": False,
}
temporary = destination.with_name(destination.name + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
print(chosen["checkpoint"])
PY
)
selected_step=$(basename "$selected")
cache=$adapter_cache_root/$selected_step

ssh "$model_host" bash -s -- "$cache" "$selected" <<'REMOTE'
set -euo pipefail
cache=$1
selected=$2
mkdir -p "$(dirname "$cache")"
if [[ ! -f "$cache/zeva_adapter.pth" ]]; then
  temporary="$cache.importing"
  rm -rf -- "$temporary"
  mkdir -p "$temporary"
  cp -a "$selected"/. "$temporary"/
  mv "$temporary" "$cache"
fi
test -f "$cache/model.safetensors"
test -f "$cache/zeva_adapter.pth"
REMOTE

# Re-saving safetensors can change the 32-byte metadata/header layout. Adapter
# training is valid only if all named tensors, shapes, dtypes, and values are
# nevertheless bit-identical to the untouched PI foundation.
foundation_identity=$eval_parent/foundation_tensor_identity-$selected_step.json
ssh "$model_host" python3 - "$cache/model.safetensors" \
  "$foundation/model.safetensors" "$foundation_identity" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

candidate_path, foundation_path, destination = map(Path, sys.argv[1:])
different = []
with safe_open(candidate_path, framework="pt", device="cpu") as candidate, \
        safe_open(foundation_path, framework="pt", device="cpu") as foundation:
    candidate_keys = set(candidate.keys())
    foundation_keys = set(foundation.keys())
    for key in sorted(candidate_keys & foundation_keys):
        left = candidate.get_tensor(key)
        right = foundation.get_tensor(key)
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            different.append(key)
    payload = {
        "schema": "zeva-frozen-foundation-tensor-identity-v1",
        "candidate": str(candidate_path),
        "foundation": str(foundation_path),
        "candidate_bytes": candidate_path.stat().st_size,
        "foundation_bytes": foundation_path.stat().st_size,
        "candidate_tensor_count": len(candidate_keys),
        "foundation_tensor_count": len(foundation_keys),
        "missing_from_candidate": sorted(foundation_keys - candidate_keys),
        "extra_in_candidate": sorted(candidate_keys - foundation_keys),
        "different_tensors": different,
        "tensor_values_bit_identical": (
            candidate_keys == foundation_keys and not different
        ),
    }
if not payload["tensor_values_bit_identical"]:
    raise RuntimeError(f"frozen PI tensor identity failed: {payload}")
temporary = destination.with_name(destination.name + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY

zeva_config=$eval_parent/configs/zeva-$selected_step.yml
python3 - "$zeva_config" "$cache" "$foundation" "$goal_embedding" "$stage1_root" <<'PY'
import sys
from pathlib import Path

destination, checkpoint, foundation, goal, stage1 = sys.argv[1:]
Path(destination).write_text(f"""policy_name: zeva_policy
handoff_root: /mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
foundation_checkpoint: {foundation}
goal_embedding_checkpoint: {goal}
stage2_checkpoint: {checkpoint}
zte_checkpoint: {stage1}/stage1-zte/zte_best.pth
causal_bank: {stage1}/stage1-zte/train_causal_bank.pt
retrieval_checkpoint: {stage1}/stage1.5-task-retrieval/task_retrieval.pth
baseline_only: false
device: cuda
model_rng_seed: 20260907
""")
PY

model_ip=${MODEL_IP:-$(ssh "$model_host" "hostname -I" | tr ' ' '\n' | grep '^172\.' | head -1)}
env \
  MODEL_HOST="$model_host" RENDER_HOST="$render_host" MODEL_IP="$model_ip" \
  OUTPUT_ROOT="$output" \
  BASELINE_CONFIG="$historical_root/../checkpoint-gates/b1000-z250-seeded-v1/configs/anchor.yml" \
  ZEVA_CONFIG="$zeva_config" ANCHOR_CONFIG="" \
  BASELINE_IS_UNTOUCHED_ANCHOR=true \
  PRECOMPUTED_BASELINE_ROOT="$historical_root/anchor" \
  FROZEN_SEED_MANIFEST="$historical_root/seed_manifest.json" \
  TASK_MANIFEST="$zeva_root/configs/robotwin_zeva_advantage10.json" \
  BASELINE_LABEL="untouched-pi05-best-v1-preregistered" \
  ZEVA_LABEL="frozen-pi-adapter-$selected_step" \
  MIN_BASELINE_SUCCESS_RATE=0.57 \
  bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

python3 "$zeva_root/scripts/robotwin_eval/audit_advantage10_formal.py" \
  "$output" --require-accepted
