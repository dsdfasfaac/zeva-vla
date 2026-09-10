#!/usr/bin/env bash
set -euo pipefail

# Confirmatory seed-1000 evaluation for the fixed v11 step-500 checkpoint.
# The immutable untouched Base is imported; ZeVA alone replays its exact
# expert-valid (seed, instruction) manifest after the four-cell gate passes.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
step=${ZEVA_STEP:-500}
train_root=${TRAIN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-prior-action-expert-v11/zeva}
checkpoint=$train_root/$(printf '%06d' "$step")
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-prior-action-expert-v11/step-$(printf '%06d' "$step")}
validation_summary=${VALIDATION_SUMMARY:-$eval_root/four_cell_validation_summary.json}
output=${OUTPUT_ROOT:-$eval_root/final-seed1000-precomputed-base}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
foundation=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
foundation_sha256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
baseline_source=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v5-multisplit/formal-calibrated-v5/baseline
baseline_report_sha256=fb4309cceb9529959e6f1411c362d85109d6b690af2edb54233394b549609d7d
seed_source=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v5-multisplit/formal-calibrated-v5/seed_manifest.json
seed_source_sha256=3338f043162a4ae231aa19ce27294025b2074c35819a83f92f0835156e7a9f54
model_host=${MODEL_HOST:-aigc24}
model_ip=${MODEL_IP:-172.16.80.158}
render_host=${RENDER_HOST:-aigc24}
render_runtime=${RENDER_RUNTIME:-/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin}
model_cache_root=${MODEL_CACHE_ROOT:-/data1/dingxin/zeva-eval-cache/prior-action-expert-v11}
render_vulkan_icd=${RENDER_VULKAN_ICD:-$render_runtime/.venv_robotwin/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json}
render_ld_library_path=${RENDER_LD_LIBRARY_PATH:-/data2/dingxin:/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/lib}

for path in "$checkpoint/model.safetensors" "$checkpoint/zeva_adapter.pth" \
  "$checkpoint/checkpoint_audit.json" "$checkpoint/residual_branch_audit.json" \
  "$validation_summary" "$task_manifest"; do
  test -s "$path"
done
test "$(sha256sum "$foundation/model.safetensors" | awk '{print $1}')" = "$foundation_sha256"
test "$(sha256sum "$baseline_source/report.json" | awk '{print $1}')" = "$baseline_report_sha256"
test "$(sha256sum "$seed_source" | awk '{print $1}')" = "$seed_source_sha256"

python3 - "$validation_summary" "$checkpoint/checkpoint_audit.json" \
  "$checkpoint/residual_branch_audit.json" <<'PY'
import json, sys
validation, checkpoint, residual = (json.load(open(path)) for path in sys.argv[1:])
if validation.get("schema") != "zeva-robotwin-v11-four-cell-validation-v1":
    raise SystemExit("unexpected v11 four-cell validation schema")
if validation.get("accepted_for_seed1000_formal_evaluation") is not True:
    raise SystemExit("v11 did not pass the four-cell validation gate")
if validation.get("total_paired_episodes") != 320 or validation.get("total_gain", -1) < 12:
    raise SystemExit("v11 four-cell validation is incomplete or below its gain gate")
if validation.get("negative_cells") or validation.get("negative_tasks"):
    raise SystemExit("v11 four-cell validation contains a regression")
if checkpoint.get("passed") is not True or checkpoint["foundation_drift"]["frozen_changed_count"] != 0:
    raise SystemExit("v11 checkpoint identity audit failed")
aggregate = residual["aggregate"]
if aggregate["prior_injection_horizon"] != 15 or aggregate["injected_context_residual_rms"] != 0:
    raise SystemExit("v11 residual contract is not H15 prior-only")
PY

model_sha256=$(sha256sum "$checkpoint/model.safetensors" | awk '{print $1}')
adapter_sha256=$(sha256sum "$checkpoint/zeva_adapter.pth" | awk '{print $1}')
zeva_cache=$model_cache_root/step-${step}-${model_sha256}

cache_file() {
  local source=$1 destination=$2 expected=$3
  ssh "$model_host" "set -euo pipefail
    mkdir -p '$(dirname "$destination")'
    if [[ -s '$destination' ]]; then
      test \"\$(sha256sum '$destination' | awk '{print \$1}')\" = '$expected'
    else
      temporary='$destination.partial'
      cp '$source' \"\$temporary\"
      test \"\$(sha256sum \"\$temporary\" | awk '{print \$1}')\" = '$expected'
      mv \"\$temporary\" '$destination'
    fi"
}
cache_file "$checkpoint/model.safetensors" "$zeva_cache/model.safetensors" "$model_sha256"
cache_file "$checkpoint/zeva_adapter.pth" "$zeva_cache/zeva_adapter.pth" "$adapter_sha256"

mkdir -p "$eval_root/configs" "$output"
base_config=$eval_root/configs/final-base-untouched-best-v1.yml
zeva_config=$eval_root/configs/final-zeva-step-${step}.yml
python3 - "$base_config" "$zeva_config" "$foundation" "$zeva_cache" <<'PY'
import os, sys
from pathlib import Path
base_path, zeva_path, foundation, checkpoint = sys.argv[1:]
shared = """policy_name: zeva_policy
handoff_root: /mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
foundation_checkpoint: /mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
"""
base = shared + f"""stage2_checkpoint: {foundation}
baseline_only: true
device: cuda
model_rng_seed: 20260907
"""
zeva = shared + f"""goal_embedding_checkpoint: /mnt/100T/users/dingxin/VLA/runtime/pretrained_model-stage1-language-v1
stage2_checkpoint: {checkpoint}
zte_checkpoint: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/zte_best.pth
causal_bank: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/train_causal_bank.pt
retrieval_checkpoint: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1.5-task-retrieval/task_retrieval.pth
baseline_only: false
device: cuda
model_rng_seed: 20260907
"""
for destination, value in ((Path(base_path), base), (Path(zeva_path), zeva)):
    if destination.is_file() and destination.read_text() != value:
        raise RuntimeError(f"refusing to mutate frozen final config: {destination}")
    if not destination.is_file():
        temporary = destination.with_name(destination.name + ".partial")
        temporary.write_text(value)
        os.replace(temporary, destination)
PY

python3 - "$output/final_plan.json" "$validation_summary" "$checkpoint" \
  "$model_sha256" "$adapter_sha256" "$baseline_report_sha256" "$seed_source_sha256" <<'PY'
import json, os, sys
from pathlib import Path
destination, validation, checkpoint, model_hash, adapter_hash, baseline_hash, seed_hash = sys.argv[1:]
payload = {
    "schema": "zeva-robotwin-v11-confirmatory-final-plan-v1",
    "checkpoint": str(Path(checkpoint).resolve()),
    "checkpoint_step": 500,
    "checkpoint_selection": "offline-screened-step500-fixed-before-closed-loop",
    "closed_loop_validation": str(Path(validation).resolve()),
    "closed_loop_validation_cells": 4,
    "final_metrics_used_for_selection": False,
    "base_role": "untouched_best_v1_precomputed_anchor",
    "base_successes": 114,
    "base_episodes": 200,
    "base_report_sha256": baseline_hash,
    "seed_manifest_sha256": seed_hash,
    "absolute_start_seed": 1000,
    "episodes_per_task": 20,
    "task_count": 10,
    "model_sha256": model_hash,
    "adapter_sha256": adapter_hash,
    "acceptance": "Base>=57% and ZeVA>Base on exact same seed/instruction pairs",
}
path = Path(destination)
if path.is_file() and json.loads(path.read_text()) != payload:
    raise RuntimeError("existing v11 final plan differs; refusing mutation")
if not path.is_file():
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
PY

env MODEL_HOST="$model_host" MODEL_IP="$model_ip" BASE_PORT=19200 \
  RENDER_HOST="$render_host" RENDER_RUNTIME="$render_runtime" \
  RENDER_VULKAN_ICD="$render_vulkan_icd" RENDER_LD_LIBRARY_PATH="$render_ld_library_path" \
  RENDER_MPS_PIPE_DIRECTORY=/tmp/zeva-v11-final-mps \
  RENDER_WARP_CACHE_ROOT=/tmp/zeva-v11-final-warp \
  OUTPUT_ROOT="$output" BASELINE_CONFIG="$base_config" ZEVA_CONFIG="$zeva_config" \
  ANCHOR_CONFIG="" BASELINE_IS_UNTOUCHED_ANCHOR=true \
  PRECOMPUTED_BASELINE_ROOT="$baseline_source" PRECOMPUTED_BASELINE_EXPECTED_SUCCESSES=114 \
  FROZEN_SEED_MANIFEST="$seed_source" REQUIRE_EXPLICIT_FOUNDATION=true \
  FOUNDATION_MODEL_SHA256="$foundation_sha256" TASK_MANIFEST="$task_manifest" \
  EPISODES=20 ABSOLUTE_START_SEED=1000 MODEL_RNG_SEED=20260907 \
  MODEL_SEED_POLICY=continuous MIN_BASELINE_SUCCESS_RATE=0.57 \
  BASELINE_LABEL=v11-untouched-best-v1-seed1000 \
  ZEVA_LABEL=v11-prior-action-expert-step-${step}-seed1000 \
  bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh" \
  > "$output/launcher.log" 2>&1

python3 "$zeva_root/scripts/robotwin_eval/audit_advantage10_formal.py" "$output" --require-accepted
python3 "$zeva_root/scripts/robotwin_eval/render_advantage10_final_markdown.py" \
  "$output" --output "$output/final_result.md"
