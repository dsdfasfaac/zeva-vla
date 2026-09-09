#!/usr/bin/env bash
set -euo pipefail

# Reserved seed-10000 final.  This is unreachable unless both disjoint v9
# validation splits pass.  Anchor, trained Base and ZeVA use the same frozen
# seeds/instructions and each episode must produce a video.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-anchored-v9/joint}
validation_summary=${VALIDATION_SUMMARY:-$eval_root/validation_summary.json}
validation_plan=${VALIDATION_PLAN:-$eval_root/validation_plan.json}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
final_root=${FINAL_ROOT:-$eval_root/fresh-final-seed10000}
anchor_config=${ANCHOR_CONFIG:-$zeva_root/scripts/robotwin_eval/baseline_bestv1_model_config.yml}
foundation_checkpoint=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
foundation_sha256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
model_host=${MODEL_HOST:-aigc29}
model_ip=${MODEL_IP:-172.16.80.163}
render_host=${RENDER_HOST:-aigc24}
render_runtime=${RENDER_RUNTIME:-/data1/dingxin/robotwin-formal-eval/RoboTwin}

test -s "$validation_summary"
test -s "$validation_plan"
test -s "$anchor_config"
test "$(sha256sum "$foundation_checkpoint/model.safetensors" | awk '{print $1}')" = "$foundation_sha256"
readarray -t selected < <(python3 - "$validation_summary" "$validation_plan" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
plan = json.load(open(sys.argv[2], encoding="utf-8"))
if summary.get("schema") != "zeva-robotwin-anchored-v9-closed-loop-validation-summary-v1":
    raise SystemExit("unexpected anchored-v9 validation-summary schema")
if summary.get("accepted_for_fresh_final_test") is not True:
    raise SystemExit("fresh final is forbidden because anchored-v9 validation did not pass")
if plan.get("schema") != "zeva-robotwin-anchored-v9-closed-loop-validation-plan-v1":
    raise SystemExit("unexpected anchored-v9 validation-plan schema")
if plan.get("closed_loop_metrics_used_for_checkpoint_selection") is not False:
    raise SystemExit("checkpoint selection was contaminated by closed-loop metrics")
for key in ("base_checkpoint", "zeva_checkpoint", "base_checkpoint_step",
            "zeva_checkpoint_step", "base_model_sha256", "zeva_model_sha256",
            "zeva_adapter_sha256", "base_config", "zeva_config",
            "base_config_sha256", "zeva_config_sha256"):
    print(plan[key])
PY
)
base_checkpoint=${selected[0]}
zeva_checkpoint=${selected[1]}
base_step=${selected[2]}
zeva_step=${selected[3]}
base_model_sha256=${selected[4]}
zeva_model_sha256=${selected[5]}
zeva_adapter_sha256=${selected[6]}
base_config=${selected[7]}
zeva_config=${selected[8]}
base_config_sha256=${selected[9]}
zeva_config_sha256=${selected[10]}
test "$(sha256sum "$base_checkpoint/model.safetensors" | awk '{print $1}')" = "$base_model_sha256"
test "$(sha256sum "$zeva_checkpoint/model.safetensors" | awk '{print $1}')" = "$zeva_model_sha256"
test "$(sha256sum "$zeva_checkpoint/zeva_adapter.pth" | awk '{print $1}')" = "$zeva_adapter_sha256"
test "$(sha256sum "$base_config" | awk '{print $1}')" = "$base_config_sha256"
test "$(sha256sum "$zeva_config" | awk '{print $1}')" = "$zeva_config_sha256"
mkdir -p "$final_root"

python3 - "$final_root/final_plan.json" "$base_checkpoint" "$zeva_checkpoint" \
  "$base_step" "$zeva_step" "$base_model_sha256" "$zeva_model_sha256" \
  "$zeva_adapter_sha256" "$base_config" "$zeva_config" "$anchor_config" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(destination, base_checkpoint, zeva_checkpoint, base_step, zeva_step, base_hash,
 zeva_hash, adapter_hash, base_config, zeva_config, anchor_config) = sys.argv[1:]
payload = {
    "schema": "zeva-robotwin-anchored-v9-fresh-final-plan-v1",
    "base_checkpoint_step": int(base_step),
    "zeva_checkpoint_step": int(zeva_step),
    "base_checkpoint": str(Path(base_checkpoint).resolve()),
    "zeva_checkpoint": str(Path(zeva_checkpoint).resolve()),
    "base_model_sha256": base_hash,
    "zeva_model_sha256": zeva_hash,
    "zeva_adapter_sha256": adapter_hash,
    "base_config": str(Path(base_config).resolve()),
    "zeva_config": str(Path(zeva_config).resolve()),
    "anchor_config": str(Path(anchor_config).resolve()),
    "anchor_config_sha256": hashlib.sha256(Path(anchor_config).read_bytes()).hexdigest(),
    "checkpoint_selected_with": "train95_validation5_only",
    "closed_loop_validation_passed_before_final": True,
    "final_metrics_used_for_training_or_selection": False,
    "task_count": 10,
    "episodes_per_task": 20,
    "absolute_start_seed": 10000,
    "minimum_baseline_success_rate": 0.57,
    "acceptance": "trained Base>=max(same-seed untouched PI,57%) and ZeVA>trained Base",
}
path = Path(destination)
if path.is_file() and json.loads(path.read_text()) != payload:
    raise RuntimeError("existing anchored-v9 final plan differs; refusing mutation")
if not path.is_file():
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
PY

if [[ ! -s "$final_root/paired_report.json" ]]; then
  env MODEL_HOST="$model_host" MODEL_IP="$model_ip" \
    RENDER_HOST="$render_host" RENDER_RUNTIME="$render_runtime" \
    OUTPUT_ROOT="$final_root" BASELINE_CONFIG="$base_config" ZEVA_CONFIG="$zeva_config" \
    ANCHOR_CONFIG="$anchor_config" BASELINE_IS_UNTOUCHED_ANCHOR=false \
    REQUIRE_EXPLICIT_FOUNDATION=true FOUNDATION_MODEL_SHA256="$foundation_sha256" \
    TASK_MANIFEST="$task_manifest" EPISODES=20 ABSOLUTE_START_SEED=10000 \
    MIN_BASELINE_SUCCESS_RATE=0.57 \
    BASELINE_LABEL="anchored-v9-base-step-$base_step-fresh-final" \
    ANCHOR_LABEL="untouched-best-v1-same-seed-anchor" \
    ZEVA_LABEL="anchored-v9-zeva-step-$zeva_step-fresh-final" \
    bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"
fi

python3 - "$final_root/seed_manifest.json" \
  "$eval_root/split-c/seed_manifest.json" "$eval_root/split-d/seed_manifest.json" <<'PY'
import json
import sys

manifests = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]
names = ("final", "validation-c", "validation-d")
for left in range(len(manifests)):
    for right in range(left + 1, len(manifests)):
        for task, rows in manifests[left]["tasks"].items():
            a = {int(row["seed"]) for row in rows}
            b = {int(row["seed"]) for row in manifests[right]["tasks"][task]}
            overlap = sorted(a & b)
            if overlap:
                raise RuntimeError(f"{names[left]}/{names[right]} seed leakage for {task}: {overlap}")
print("anchored-v9 final seed manifest is disjoint from both validation splits")
PY

python3 "$zeva_root/scripts/robotwin_eval/audit_advantage10_formal.py" \
  "$final_root" --require-accepted
