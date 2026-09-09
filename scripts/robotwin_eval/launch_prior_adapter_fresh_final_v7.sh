#!/usr/bin/env bash
set -euo pipefail

# One-shot fresh final comparison.  This launcher refuses to run until both
# new closed-loop validation splits approve the immutable checkpoint.  The
# seed-10000 final manifest is never used for training, checkpoint selection,
# residual scaling, or task gating.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-prior-only-v7}
validation_summary=${VALIDATION_SUMMARY:-$eval_root/validation_summary.json}
validation_plan=${VALIDATION_PLAN:-$eval_root/validation_plan.json}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
final_root=${FINAL_ROOT:-$eval_root/fresh-final-seed10000}
baseline_config=${BASELINE_CONFIG:-$zeva_root/scripts/robotwin_eval/baseline_bestv1_model_config.yml}
foundation_checkpoint=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
foundation_sha256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
model_host=${MODEL_HOST:-aigc29}
model_ip=${MODEL_IP:-172.16.80.163}
render_host=${RENDER_HOST:-aigc24}
render_runtime=${RENDER_RUNTIME:-/data1/dingxin/robotwin-formal-eval/RoboTwin}

test -s "$validation_summary"
test -s "$validation_plan"
test -s "$baseline_config"
grep -Fqx "foundation_checkpoint: $foundation_checkpoint" "$baseline_config"
test "$(sha256sum "$foundation_checkpoint/model.safetensors" | awk '{print $1}')" = "$foundation_sha256"
readarray -t selected < <(python3 - "$validation_summary" "$validation_plan" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
plan = json.load(open(sys.argv[2], encoding="utf-8"))
if summary.get("schema") != "zeva-robotwin-v7-fresh-closed-loop-validation-summary-v1":
    raise SystemExit("unexpected validation-summary schema")
if summary.get("accepted_for_fresh_final_test") is not True:
    raise SystemExit("fresh final is forbidden because closed-loop validation did not pass")
if plan.get("closed_loop_metrics_used_for_checkpoint_selection") is not False:
    raise SystemExit("checkpoint selection was contaminated by closed-loop metrics")
print(plan["checkpoint"])
print(plan["checkpoint_step"])
print(plan["adapter_sha256"])
PY
)
checkpoint=${selected[0]}
checkpoint_step=${selected[1]}
adapter_sha256=${selected[2]}
test -s "$checkpoint/model.safetensors"
test -s "$checkpoint/zeva_adapter.pth"
test "$(sha256sum "$checkpoint/zeva_adapter.pth" | awk '{print $1}')" = "$adapter_sha256"
mkdir -p "$final_root"

python3 - "$final_root/final_plan.json" "$checkpoint" "$checkpoint_step" "$adapter_sha256" <<'PY'
import json
import os
import sys
from pathlib import Path

destination, checkpoint, step, adapter_hash = sys.argv[1:]
payload = {
    "schema": "zeva-robotwin-v7-fresh-final-plan-v1",
    "checkpoint": checkpoint,
    "checkpoint_step": int(step),
    "adapter_sha256": adapter_hash,
    "checkpoint_selected_with": "train95_validation5_only",
    "closed_loop_validation_passed_before_final": True,
    "final_metrics_used_for_training_or_selection": False,
    "task_count": 10,
    "episodes_per_task": 20,
    "absolute_start_seed": 10000,
    "minimum_baseline_success_rate": 0.57,
    "acceptance": "Base>=57% and ZeVA>Base",
}
path = Path(destination)
if path.is_file():
    if json.loads(path.read_text()) != payload:
        raise RuntimeError("existing final plan differs; refusing to mutate a started final test")
else:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
PY

config=$eval_root/configs/final-step-${checkpoint_step}.yml
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
""", encoding="utf-8")
PY

if [[ ! -s "$final_root/paired_report.json" ]]; then
  env MODEL_HOST="$model_host" MODEL_IP="$model_ip" \
    RENDER_HOST="$render_host" RENDER_RUNTIME="$render_runtime" \
    OUTPUT_ROOT="$final_root" \
    BASELINE_CONFIG="$baseline_config" \
    REQUIRE_EXPLICIT_FOUNDATION=true FOUNDATION_MODEL_SHA256="$foundation_sha256" \
    ZEVA_CONFIG="$config" ANCHOR_CONFIG="" BASELINE_IS_UNTOUCHED_ANCHOR=true \
    TASK_MANIFEST="$task_manifest" EPISODES=20 ABSOLUTE_START_SEED=10000 \
    MIN_BASELINE_SUCCESS_RATE=0.57 \
    BASELINE_LABEL="fresh-final-seed10000-untouched-pi05" \
    ZEVA_LABEL="prior-adapter-v7-step-$checkpoint_step-fresh-final" \
    bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"
fi

python3 - "$final_root/seed_manifest.json" \
  "$eval_root/split-a/seed_manifest.json" "$eval_root/split-b/seed_manifest.json" <<'PY'
import json
import sys

manifests = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]
names = ("final", "validation-a", "validation-b")
for left in range(len(manifests)):
    for right in range(left + 1, len(manifests)):
        for task, rows in manifests[left]["tasks"].items():
            a = {int(row["seed"]) for row in rows}
            b = {int(row["seed"]) for row in manifests[right]["tasks"][task]}
            overlap = sorted(a & b)
            if overlap:
                raise RuntimeError(f"{names[left]}/{names[right]} seed leakage for {task}: {overlap}")
print("fresh final seed manifest is pairwise disjoint from both validation splits")
PY

python3 "$zeva_root/scripts/robotwin_eval/audit_advantage10_formal.py" \
  "$final_root" --require-accepted
