#!/usr/bin/env bash
set -euo pipefail

# Evaluate the validation5-selected v7 checkpoint on two new, mutually
# disjoint closed-loop seed streams.  These runs are calibration evidence only;
# the later final stream begins at seed 10000 and is never read here.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
train_root=${TRAIN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-prior-only-v7-corrected/zeva}
selection=${SELECTION:-$train_root/deployment_selection.json}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-prior-only-v7}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
episodes=${EPISODES:-8}
baseline_config=${BASELINE_CONFIG:-$zeva_root/scripts/robotwin_eval/baseline_bestv1_model_config.yml}
foundation_checkpoint=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
foundation_sha256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe

model_host_a=${MODEL_HOST_A:-aigc29}
model_ip_a=${MODEL_IP_A:-172.16.80.163}
render_host_a=${RENDER_HOST_A:-aigc24}
render_runtime_a=${RENDER_RUNTIME_A:-/data1/dingxin/robotwin-formal-eval/RoboTwin}
model_host_b=${MODEL_HOST_B:-aigc32}
model_ip_b=${MODEL_IP_B:-172.16.80.166}
render_host_b=${RENDER_HOST_B:-aigc15}
render_runtime_b=${RENDER_RUNTIME_B:-/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin}

test -s "$selection"
test -s "$task_manifest"
test -s "$baseline_config"
grep -Fqx "foundation_checkpoint: $foundation_checkpoint" "$baseline_config"
test "$(sha256sum "$foundation_checkpoint/model.safetensors" | awk '{print $1}')" = "$foundation_sha256"
mkdir -p "$eval_root/configs"

readarray -t selected < <(python3 - "$selection" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("schema") != "zeva-robotwin-prior-adapter-validation-selection-v1":
    raise SystemExit("unexpected checkpoint-selection schema")
if payload.get("formal_or_closed_loop_test_metrics_used") is not False:
    raise SystemExit("checkpoint selection must be validation5-only")
selected = payload.get("selected")
if not selected:
    raise SystemExit("no checkpoint passed the offline non-regression gate")
print(selected["checkpoint"])
print(selected["step"])
print(selected["artifact_identity"]["adapter_sha256"])
print(selected["artifact_identity"]["model_sha256"])
PY
)
checkpoint=${selected[0]}
checkpoint_step=${selected[1]}
adapter_sha256=${selected[2]}
model_sha256=${selected[3]}
test -s "$checkpoint/model.safetensors"
test -s "$checkpoint/zeva_adapter.pth"
test "$(sha256sum "$checkpoint/zeva_adapter.pth" | awk '{print $1}')" = "$adapter_sha256"
test "$(sha256sum "$checkpoint/model.safetensors" | awk '{print $1}')" = "$model_sha256"
foundation_identity=$eval_root/foundation_identity-step-${checkpoint_step}.json
python3 "$zeva_root/scripts/verify_frozen_safetensors_identity.py" \
  "$checkpoint/model.safetensors" "$foundation_checkpoint/model.safetensors" \
  "$foundation_identity"

config=$eval_root/configs/selected-step-${checkpoint_step}.yml
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

cat_plan=$eval_root/validation_plan.json
python3 - "$cat_plan" "$selection" "$checkpoint" "$checkpoint_step" "$adapter_sha256" "$model_sha256" "$foundation_identity" "$episodes" <<'PY'
import json
import os
import sys
from pathlib import Path

destination, selection, checkpoint, step, adapter_hash, model_hash, foundation_identity, episodes = sys.argv[1:]
payload = {
    "schema": "zeva-robotwin-v7-fresh-closed-loop-validation-plan-v1",
    "selection": selection,
    "selection_data": "train95_validation5_only",
    "closed_loop_metrics_used_for_checkpoint_selection": False,
    "checkpoint": checkpoint,
    "checkpoint_step": int(step),
    "adapter_sha256": adapter_hash,
    "model_sha256": model_hash,
    "foundation_tensor_identity": foundation_identity,
    "episodes_per_task_per_split": int(episodes),
    "splits": {"a": {"start_seed": 5000}, "b": {"start_seed": 6000}},
    "reserved_final_start_seed": 10000,
    "acceptance": "delta>=0 on each split and combined ZeVA-Base gain>=6/160",
}
temporary = Path(destination + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY

run_split() {
  local name=$1
  local start_seed=$2
  local model_host=$3
  local model_ip=$4
  local render_host=$5
  local render_runtime=$6
  local output=$eval_root/$name
  if [[ -s "$output/paired_report.json" ]]; then
    return
  fi
  env MODEL_HOST="$model_host" MODEL_IP="$model_ip" \
    RENDER_HOST="$render_host" RENDER_RUNTIME="$render_runtime" \
    OUTPUT_ROOT="$output" \
    BASELINE_CONFIG="$baseline_config" \
    REQUIRE_EXPLICIT_FOUNDATION=true FOUNDATION_MODEL_SHA256="$foundation_sha256" \
    ZEVA_CONFIG="$config" ANCHOR_CONFIG="" BASELINE_IS_UNTOUCHED_ANCHOR=true \
    TASK_MANIFEST="$task_manifest" EPISODES="$episodes" \
    ABSOLUTE_START_SEED="$start_seed" MIN_BASELINE_SUCCESS_RATE=0 \
    BASELINE_LABEL="fresh-validation-$name-untouched-pi05" \
    ZEVA_LABEL="prior-adapter-v7-step-$checkpoint_step-$name" \
    bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh" \
    > "$eval_root/$name.launcher.log" 2>&1
}

run_split split-a 5000 "$model_host_a" "$model_ip_a" "$render_host_a" "$render_runtime_a" &
pid_a=$!
run_split split-b 6000 "$model_host_b" "$model_ip_b" "$render_host_b" "$render_runtime_b" &
pid_b=$!
printf '%s\n' "$pid_a" > "$eval_root/split-a.pid"
printf '%s\n' "$pid_b" > "$eval_root/split-b.pid"
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
if (( status != 0 )); then
  exit "$status"
fi

python3 - "$eval_root" "$episodes" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
episodes = int(sys.argv[2])
reports = {name: json.loads((root / name / "paired_report.json").read_text())
           for name in ("split-a", "split-b")}
seed_sets = {}
for name in reports:
    manifest = json.loads((root / name / "seed_manifest.json").read_text())
    seed_sets[name] = {task: {int(row["seed"]) for row in rows}
                       for task, rows in manifest["tasks"].items()}
overlaps = {task: sorted(seed_sets["split-a"][task] & seed_sets["split-b"][task])
            for task in seed_sets["split-a"]}
if any(overlaps.values()):
    raise RuntimeError(f"validation splits overlap: {overlaps}")
rows = []
for name, report in reports.items():
    delta_count = round(report["absolute_delta"] * report["total_paired_episodes"])
    rows.append({"split": name,
                 "baseline_success_rate": report["baseline_success_rate"],
                 "zeva_success_rate": report["zeva_success_rate"],
                 "delta_count": delta_count,
                 "episodes": report["total_paired_episodes"]})
combined_delta = sum(row["delta_count"] for row in rows)
accepted = all(row["delta_count"] >= 0 for row in rows) and combined_delta >= 6
payload = {
    "schema": "zeva-robotwin-v7-fresh-closed-loop-validation-summary-v1",
    "accepted_for_fresh_final_test": accepted,
    "criteria": {"each_split_delta_nonnegative": True,
                 "minimum_combined_delta_count": 6},
    "splits_pairwise_disjoint": True,
    "rows": rows,
    "combined_delta_count": combined_delta,
    "combined_episodes_per_condition": sum(row["episodes"] for row in rows),
    "next_action": "launch_fresh_final" if accepted else "diagnose_and_retrain",
}
destination = root / "validation_summary.json"
temporary = destination.with_name(destination.name + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
print(json.dumps(payload, indent=2))
PY
