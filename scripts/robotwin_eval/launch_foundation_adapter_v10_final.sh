#!/usr/bin/env bash
set -euo pipefail

# Confirmatory seed-1000 evaluation for the v10 foundation adapter.  The
# untouched best-v1 Base was evaluated and video-audited before this adapter
# existed, so we import that immutable rollout and execute ZeVA only on the
# exact same 10x20 seed/instruction pairs.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
train_root=${TRAIN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-foundation-adapter-v10/adapter}
selection=${SELECTION:-$(dirname "$train_root")/$(basename "$train_root")-deployment-selection.json}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-foundation-adapter-v10}
output=${OUTPUT_ROOT:-$eval_root/final-seed1000-precomputed-base}
validation_summary=${VALIDATION_SUMMARY:-$eval_root/validation_summary.json}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
foundation=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
foundation_sha256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
baseline_source=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v5-multisplit/formal-calibrated-v5/baseline
baseline_report_sha256=fb4309cceb9529959e6f1411c362d85109d6b690af2edb54233394b549609d7d
seed_source=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v5-multisplit/formal-calibrated-v5/seed_manifest.json
seed_source_sha256=3338f043162a4ae231aa19ce27294025b2074c35819a83f92f0835156e7a9f54
model_host=${MODEL_HOST:-aigc29}
model_ip=${MODEL_IP:-172.16.80.163}
render_host=${RENDER_HOST:-aigc14}
render_runtime=${RENDER_RUNTIME:-/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin}
model_cache_root=${MODEL_CACHE_ROOT:-/data1/dingxin/zeva-eval-cache/foundation-adapter-v10-final}

test -s "$train_root/COMPLETE"
test -s "$selection"
test -s "$validation_summary"
test -s "$task_manifest"
test "$(sha256sum "$foundation/model.safetensors" | awk '{print $1}')" = "$foundation_sha256"
test "$(sha256sum "$baseline_source/report.json" | awk '{print $1}')" = "$baseline_report_sha256"
test "$(sha256sum "$seed_source" | awk '{print $1}')" = "$seed_source_sha256"

readarray -t selected < <(python3 - "$selection" "$validation_summary" "$foundation" <<'PY'
import json
import sys
from pathlib import Path

selection = json.load(open(sys.argv[1], encoding="utf-8"))
validation = json.load(open(sys.argv[2], encoding="utf-8"))
foundation = Path(sys.argv[3]).resolve()
if selection.get("schema") != "zeva-robotwin-foundation-adapter-v10-validation-selection-v1":
    raise SystemExit("unexpected v10 selection schema")
if selection.get("selection_data") != "train95_validation5_only":
    raise SystemExit("v10 checkpoint was not selected on validation5 only")
if selection.get("closed_loop_metrics_used") is not False:
    raise SystemExit("closed-loop results contaminated checkpoint selection")
if Path(selection.get("fixed_base_checkpoint", "/")).resolve() != foundation:
    raise SystemExit("v10 selection used the wrong untouched Base")
if validation.get("schema") != "zeva-robotwin-foundation-adapter-v10-closed-loop-validation-summary-v1":
    raise SystemExit("unexpected v10 closed-loop validation schema")
if validation.get("accepted_for_fresh_final_test") is not True:
    raise SystemExit("v10 did not pass the preregistered closed-loop validation gate")
if validation.get("combined_episodes_per_condition") != 160:
    raise SystemExit("v10 closed-loop validation is incomplete")
row = selection.get("selected")
if not row or row.get("eligible") is not True:
    raise SystemExit("no eligible v10 checkpoint")
artifacts = row["artifacts"]
print(row["zeva_checkpoint"])
print(row["step"])
print(artifacts["base_model_sha256"])
print(artifacts["zeva_model_sha256"])
print(artifacts["zeva_adapter_sha256"])
PY
)
zeva_checkpoint=${selected[0]}
zeva_step=${selected[1]}
base_model_sha256=${selected[2]}
zeva_model_sha256=${selected[3]}
zeva_adapter_sha256=${selected[4]}
test "$base_model_sha256" = "$foundation_sha256"
test "$zeva_model_sha256" = "$foundation_sha256"
test "$(sha256sum "$zeva_checkpoint/model.safetensors" | awk '{print $1}')" = "$zeva_model_sha256"
test "$(sha256sum "$zeva_checkpoint/zeva_adapter.pth" | awk '{print $1}')" = "$zeva_adapter_sha256"

cache_file() {
  local source=$1 destination=$2 expected_sha256=$3
  ssh "$model_host" "set -euo pipefail
    mkdir -p '$(dirname "$destination")'
    if [[ -e '$destination' ]]; then
      test \"\$(sha256sum '$destination' | awk '{print \$1}')\" = '$expected_sha256'
    else
      temporary='$destination.partial'
      cp '$source' \"\$temporary\"
      test \"\$(sha256sum \"\$temporary\" | awk '{print \$1}')\" = '$expected_sha256'
      mv \"\$temporary\" '$destination'
    fi"
}
zeva_cache=$model_cache_root/zeva-${zeva_step}-${zeva_model_sha256}
cache_file "$zeva_checkpoint/model.safetensors" "$zeva_cache/model.safetensors" "$zeva_model_sha256"
cache_file "$zeva_checkpoint/zeva_adapter.pth" "$zeva_cache/zeva_adapter.pth" "$zeva_adapter_sha256"

mkdir -p "$eval_root/configs" "$output"
base_config=$eval_root/configs/final-base-untouched-best-v1.yml
zeva_config=$eval_root/configs/final-zeva-step-${zeva_step}.yml
python3 - "$base_config" "$zeva_config" "$foundation" "$zeva_cache" <<'PY'
import os
import sys
from pathlib import Path

base_path, zeva_path, foundation, zeva_checkpoint = sys.argv[1:]
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
stage2_checkpoint: {zeva_checkpoint}
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

# Freeze the confirmatory design before any v10 seed-1000 rollout is read.
python3 - "$output/final_plan.json" "$selection" "$validation_summary" \
  "$zeva_checkpoint" "$zeva_step" "$base_model_sha256" "$zeva_model_sha256" \
  "$zeva_adapter_sha256" "$baseline_report_sha256" "$seed_source_sha256" <<'PY'
import json
import os
import sys
from pathlib import Path

(destination, selection, validation, checkpoint, step, base_hash, model_hash,
 adapter_hash, baseline_hash, seed_hash) = sys.argv[1:]
payload = {
    "schema": "zeva-robotwin-foundation-adapter-v10-confirmatory-final-plan-v1",
    "checkpoint_selection": str(Path(selection).resolve()),
    "checkpoint_selection_data": "train95_validation5_only",
    "closed_loop_validation": str(Path(validation).resolve()),
    "closed_loop_validation_required_accepted": True,
    "final_metrics_used_for_selection": False,
    "base_role": "untouched_best_v1_is_anchor",
    "base_successes": 114,
    "base_episodes": 200,
    "base_report_sha256": baseline_hash,
    "seed_manifest_sha256": seed_hash,
    "absolute_start_seed": 1000,
    "episodes_per_task": 20,
    "task_count": 10,
    "zeva_checkpoint": str(Path(checkpoint).resolve()),
    "zeva_checkpoint_step": int(step),
    "base_model_sha256": base_hash,
    "zeva_model_sha256": model_hash,
    "zeva_adapter_sha256": adapter_hash,
    "acceptance": "Base>=57% and ZeVA>Base on exact same seed/instruction pairs",
}
path = Path(destination)
if path.is_file() and json.loads(path.read_text()) != payload:
    raise RuntimeError("existing confirmatory final plan differs; refusing mutation")
if not path.is_file():
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
PY

env MODEL_HOST="$model_host" MODEL_IP="$model_ip" BASE_PORT=19200 \
  RENDER_HOST="$render_host" RENDER_RUNTIME="$render_runtime" \
  RENDER_MPS_PIPE_DIRECTORY=/tmp/zeva-v10-final-mps \
  OUTPUT_ROOT="$output" BASELINE_CONFIG="$base_config" ZEVA_CONFIG="$zeva_config" \
  ANCHOR_CONFIG="" BASELINE_IS_UNTOUCHED_ANCHOR=true \
  PRECOMPUTED_BASELINE_ROOT="$baseline_source" \
  PRECOMPUTED_BASELINE_EXPECTED_SUCCESSES=114 \
  FROZEN_SEED_MANIFEST="$seed_source" \
  REQUIRE_EXPLICIT_FOUNDATION=true FOUNDATION_MODEL_SHA256="$foundation_sha256" \
  TASK_MANIFEST="$task_manifest" EPISODES=20 ABSOLUTE_START_SEED=1000 \
  MODEL_RNG_SEED=20260907 MODEL_SEED_POLICY=continuous MIN_BASELINE_SUCCESS_RATE=0.57 \
  BASELINE_LABEL=v10-untouched-best-v1-seed1000 \
  ZEVA_LABEL=v10-foundation-adapter-step-${zeva_step}-seed1000 \
  bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh" \
  > "$output/launcher.log" 2>&1

python3 "$zeva_root/scripts/robotwin_eval/audit_advantage10_formal.py" \
  "$output" --require-accepted
python3 "$zeva_root/scripts/robotwin_eval/render_advantage10_final_markdown.py" \
  "$output" --output "$output/final_result.md"

