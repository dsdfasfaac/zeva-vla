#!/usr/bin/env bash
set -euo pipefail

# Closed-loop validation for a validation5-only selected v10 adapter. Base is
# the untouched RoboTwin best-v1 policy. The two 10x8 streams are disjoint from
# all v7-v9 validation/final streams and cannot affect checkpoint selection.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
train_root=${TRAIN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-foundation-adapter-v10/adapter}
selection=${SELECTION:-$(dirname "$train_root")/$(basename "$train_root")-deployment-selection.json}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-foundation-adapter-v10}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
foundation_checkpoint=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
foundation_sha256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
episodes=${EPISODES:-8}
split_g_name=${SPLIT_G_NAME:-split-g}
split_h_name=${SPLIT_H_NAME:-split-h}
start_seed_g=${START_SEED_G:-13000}
start_seed_h=${START_SEED_H:-14000}
reserved_final_seed=${RESERVED_FINAL_SEED:-15000}
model_host_g=${MODEL_HOST_G:-aigc01}
model_ip_g=${MODEL_IP_G:-172.16.80.135}
render_host_g=${RENDER_HOST_G:-aigc29}
model_host_h=${MODEL_HOST_H:-aigc28}
model_ip_h=${MODEL_IP_H:-172.16.80.162}
render_host_h=${RENDER_HOST_H:-aigc14}
render_runtime=${RENDER_RUNTIME:-/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin}
model_cache_root=${MODEL_CACHE_ROOT:-/data1/dingxin/zeva-eval-cache/foundation-adapter-v10}

test -s "$train_root/COMPLETE"
test -s "$selection"
test -s "$task_manifest"
test "$(sha256sum "$foundation_checkpoint/model.safetensors" | awk '{print $1}')" = "$foundation_sha256"
mkdir -p "$eval_root/configs"

readarray -t selected < <(python3 - "$selection" "$foundation_checkpoint" <<'PY'
import json
import sys
from pathlib import Path

payload = json.load(open(sys.argv[1], encoding="utf-8"))
foundation = Path(sys.argv[2]).resolve()
if payload.get("schema") != "zeva-robotwin-foundation-adapter-v10-validation-selection-v1":
    raise SystemExit("unexpected v10 checkpoint-selection schema")
if payload.get("selection_data") != "train95_validation5_only":
    raise SystemExit("checkpoint selection must be validation5-only")
if payload.get("closed_loop_metrics_used") is not False:
    raise SystemExit("checkpoint selection was contaminated by closed-loop metrics")
if Path(payload.get("fixed_base_checkpoint", "/")).resolve() != foundation:
    raise SystemExit("selection used a different untouched Base")
row = payload.get("selected")
if not row or row.get("eligible") is not True:
    raise SystemExit("no v10 checkpoint passed the offline gate")
if Path(row["foundation_checkpoint"]).resolve() != foundation:
    raise SystemExit("selected row used a different untouched Base")
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
  local host=$1 source=$2 destination=$3 expected_sha256=$4
  ssh "$host" "set -euo pipefail
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
base_cache=$model_cache_root/base-${foundation_sha256}
zeva_cache=$model_cache_root/zeva-${zeva_step}-${zeva_model_sha256}
for host in "$model_host_g" "$model_host_h"; do
  cache_file "$host" "$foundation_checkpoint/model.safetensors" \
    "$base_cache/model.safetensors" "$base_model_sha256"
  cache_file "$host" "$zeva_checkpoint/model.safetensors" \
    "$zeva_cache/model.safetensors" "$zeva_model_sha256"
  cache_file "$host" "$zeva_checkpoint/zeva_adapter.pth" \
    "$zeva_cache/zeva_adapter.pth" "$zeva_adapter_sha256"
done

base_config=$eval_root/configs/base-untouched-best-v1.yml
zeva_config=$eval_root/configs/zeva-step-${zeva_step}.yml
python3 - "$base_config" "$zeva_config" "$base_cache" "$zeva_cache" <<'PY'
import os
import sys
from pathlib import Path

base_path, zeva_path, base_checkpoint, zeva_checkpoint = sys.argv[1:]
shared = """policy_name: zeva_policy
handoff_root: /mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
foundation_checkpoint: /mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
"""
base = shared + f"""stage2_checkpoint: {base_checkpoint}
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
        raise RuntimeError(f"refusing to mutate frozen rollout config: {destination}")
    if not destination.is_file():
        temporary = destination.with_name(destination.name + ".partial")
        temporary.write_text(value)
        os.replace(temporary, destination)
PY

python3 - "$eval_root/validation_plan.json" "$selection" "$zeva_checkpoint" \
  "$zeva_step" "$base_model_sha256" "$zeva_model_sha256" "$zeva_adapter_sha256" \
  "$episodes" "$start_seed_g" "$start_seed_h" "$reserved_final_seed" <<'PY'
import json
import os
import sys
from pathlib import Path

(destination, selection, checkpoint, step, base_hash, model_hash, adapter_hash,
 episodes, seed_g, seed_h, final_seed) = sys.argv[1:]
payload = {
    "schema": "zeva-robotwin-foundation-adapter-v10-closed-loop-validation-plan-v1",
    "selection": str(Path(selection).resolve()),
    "selection_data": "train95_validation5_only",
    "closed_loop_metrics_used_for_checkpoint_selection": False,
    "base_checkpoint": "/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1",
    "zeva_checkpoint": str(Path(checkpoint).resolve()),
    "zeva_checkpoint_step": int(step),
    "base_model_sha256": base_hash,
    "zeva_model_sha256": model_hash,
    "zeva_adapter_sha256": adapter_hash,
    "episodes_per_task_per_split": int(episodes),
    "splits": {"g": {"start_seed": int(seed_g)}, "h": {"start_seed": int(seed_h)}},
    "excluded_prior_seed_starts": [1000, 5000, 6000, 7000, 8000, 9000, 10000, 12000],
    "reserved_final_start_seed": int(final_seed),
    "acceptance": "delta>=0 on each split and combined ZeVA-Base gain>=6/160",
}
path = Path(destination)
if path.is_file() and json.loads(path.read_text()) != payload:
    raise RuntimeError("existing v10 validation plan differs; refusing mutation")
if not path.is_file():
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
PY

run_split() {
  local name=$1 start_seed=$2 model_host=$3 model_ip=$4 render_host=$5 port=$6
  local output=$eval_root/$name
  if [[ -s "$output/paired_report.json" ]]; then return; fi
  env MODEL_HOST="$model_host" MODEL_IP="$model_ip" BASE_PORT="$port" \
    RENDER_HOST="$render_host" RENDER_RUNTIME="$render_runtime" \
    RENDER_MPS_PIPE_DIRECTORY="/tmp/zeva-v10-${name}-mps" \
    OUTPUT_ROOT="$output" BASELINE_CONFIG="$base_config" ZEVA_CONFIG="$zeva_config" \
    ANCHOR_CONFIG="" BASELINE_IS_UNTOUCHED_ANCHOR=true \
    REQUIRE_EXPLICIT_FOUNDATION=true FOUNDATION_MODEL_SHA256="$foundation_sha256" \
    TASK_MANIFEST="$task_manifest" EPISODES="$episodes" ABSOLUTE_START_SEED="$start_seed" \
    MODEL_RNG_SEED=20260907 MODEL_SEED_POLICY=continuous MIN_BASELINE_SUCCESS_RATE=0 \
    BASELINE_LABEL="v10-untouched-best-v1-$name" \
    ZEVA_LABEL="v10-foundation-adapter-step-$zeva_step-$name" \
    bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh" \
    > "$eval_root/$name.launcher.log" 2>&1
}

run_split "$split_g_name" "$start_seed_g" "$model_host_g" "$model_ip_g" "$render_host_g" 19200 &
pid_g=$!
run_split "$split_h_name" "$start_seed_h" "$model_host_h" "$model_ip_h" "$render_host_h" 19200 &
pid_h=$!
printf '%s\n' "$pid_g" > "$eval_root/${split_g_name}.pid"
printf '%s\n' "$pid_h" > "$eval_root/${split_h_name}.pid"
status=0
wait "$pid_g" || status=1
wait "$pid_h" || status=1
(( status == 0 )) || exit "$status"

python3 "$zeva_root/scripts/robotwin_eval/audit_advantage10_formal.py" \
  "$eval_root/$split_g_name"
python3 "$zeva_root/scripts/robotwin_eval/audit_advantage10_formal.py" \
  "$eval_root/$split_h_name"

python3 - "$eval_root" "$episodes" "$split_g_name" "$split_h_name" <<'PY'
import json
import os
import sys
from pathlib import Path

root, episodes, left, right = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
reports = {name: json.loads((root / name / "paired_report.json").read_text()) for name in (left, right)}
manifests = {name: json.loads((root / name / "seed_manifest.json").read_text()) for name in reports}
for name, report in reports.items():
    if report.get("total_paired_episodes") != episodes * 10:
        raise RuntimeError(f"{name}: incomplete paired report")
for task in manifests[left]["tasks"]:
    a = {int(row["seed"]) for row in manifests[left]["tasks"][task]}
    b = {int(row["seed"]) for row in manifests[right]["tasks"][task]}
    if a & b:
        raise RuntimeError(f"v10 validation seed overlap for {task}: {sorted(a & b)}")
rows = []
for name, report in reports.items():
    rows.append({
        "split": name,
        "baseline_success_rate": report["baseline_success_rate"],
        "zeva_success_rate": report["zeva_success_rate"],
        "delta_count": round(report["absolute_delta"] * report["total_paired_episodes"]),
        "episodes_per_condition": report["total_paired_episodes"],
    })
combined = sum(row["delta_count"] for row in rows)
accepted = all(row["delta_count"] >= 0 for row in rows) and combined >= 6
payload = {
    "schema": "zeva-robotwin-foundation-adapter-v10-closed-loop-validation-summary-v1",
    "accepted_for_fresh_final_test": accepted,
    "criteria": {"each_split_delta_nonnegative": True, "minimum_combined_delta_count": 6},
    "splits_pairwise_disjoint": True,
    "rows": rows,
    "combined_delta_count": combined,
    "combined_episodes_per_condition": sum(row["episodes_per_condition"] for row in rows),
    "next_action": "launch_fresh_final" if accepted else "diagnose_and_retrain",
}
destination = root / "validation_summary.json"
temporary = destination.with_name(destination.name + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
print(json.dumps(payload, indent=2))
if not accepted:
    raise SystemExit(4)
PY
