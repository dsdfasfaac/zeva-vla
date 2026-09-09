#!/usr/bin/env bash
set -euo pipefail

# Closed-loop validation for a train95/validation5-selected anchored-v9
# checkpoint.  Base is the immutable v8 step-3000 action expert; the ZeVA
# step is selected independently without using any closed-loop result.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
candidate_root=${CANDIDATE_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-anchored-v9/joint}
base_checkpoint=${BASE_CHECKPOINT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-action-expert-v8/baseline/003000}
selection=${SELECTION:-$(dirname "$candidate_root")/$(basename "$candidate_root")-deployment-selection.json}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-anchored-v9/joint}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
episodes=${EPISODES:-8}
foundation_checkpoint=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
foundation_sha256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe

model_host_c=${MODEL_HOST_C:-aigc31}
model_ip_c=${MODEL_IP_C:-172.16.80.165}
render_host_c=${RENDER_HOST_C:-aigc24}
render_runtime_c=${RENDER_RUNTIME_C:-/data1/dingxin/robotwin-formal-eval/RoboTwin}
model_host_d=${MODEL_HOST_D:-aigc32}
model_ip_d=${MODEL_IP_D:-172.16.80.166}
render_host_d=${RENDER_HOST_D:-aigc15}
render_runtime_d=${RENDER_RUNTIME_D:-/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin}
model_cache_root=${MODEL_CACHE_ROOT:-/data1/dingxin/zeva-eval-cache/anchored-v9}

test -s "$selection"
test -s "$task_manifest"
test -s "$base_checkpoint/model.safetensors"
test "$(sha256sum "$foundation_checkpoint/model.safetensors" | awk '{print $1}')" = "$foundation_sha256"
mkdir -p "$eval_root/configs"

readarray -t selected < <(python3 - "$selection" "$base_checkpoint" <<'PY'
import json
import sys
from pathlib import Path

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected_base = Path(sys.argv[2]).resolve()
if payload.get("schema") != "zeva-robotwin-anchored-v9-validation-selection-v1":
    raise SystemExit("unexpected anchored-v9 checkpoint-selection schema")
if payload.get("selection_data") != "train95_validation5_only":
    raise SystemExit("checkpoint selection must be validation5-only")
if payload.get("closed_loop_metrics_used") is not False:
    raise SystemExit("checkpoint selection was contaminated by closed-loop metrics")
if Path(payload.get("fixed_base_checkpoint", "/")).resolve() != expected_base:
    raise SystemExit("selection used a different fixed Base checkpoint")
row = payload.get("selected")
if not row or row.get("eligible") is not True:
    raise SystemExit("no anchored-v9 checkpoint passed the offline gate")
if Path(row["base_checkpoint"]).resolve() != expected_base:
    raise SystemExit("selected row used a different Base checkpoint")
artifacts = row["artifacts"]
print(row["base_checkpoint"])
print(row["zeva_checkpoint"])
print(row["base_step"])
print(row["step"])
print(artifacts["base_model_sha256"])
print(artifacts["zeva_model_sha256"])
print(artifacts["zeva_adapter_sha256"])
PY
)
base_checkpoint=${selected[0]}
zeva_checkpoint=${selected[1]}
base_step=${selected[2]}
zeva_step=${selected[3]}
base_model_sha256=${selected[4]}
zeva_model_sha256=${selected[5]}
zeva_adapter_sha256=${selected[6]}
test "$base_checkpoint" != "$zeva_checkpoint"
test -s "$zeva_checkpoint/model.safetensors"
test -s "$zeva_checkpoint/zeva_adapter.pth"
test "$(sha256sum "$base_checkpoint/model.safetensors" | awk '{print $1}')" = "$base_model_sha256"
test "$(sha256sum "$zeva_checkpoint/model.safetensors" | awk '{print $1}')" = "$zeva_model_sha256"
test "$(sha256sum "$zeva_checkpoint/zeva_adapter.pth" | awk '{print $1}')" = "$zeva_adapter_sha256"

# Each condition starts eight model servers.  Copy each immutable artifact once
# to the model hosts' NVMe instead of making all ranks cold-read it from NFS.
# Hash-qualified cache directories make reuse safe and preserve source lineage.
base_cache=$model_cache_root/base-${base_step}-${base_model_sha256}
zeva_cache=$model_cache_root/zeva-${zeva_step}-${zeva_model_sha256}
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
for host in "$model_host_c" "$model_host_d"; do
  cache_file "$host" "$base_checkpoint/model.safetensors" \
    "$base_cache/model.safetensors" "$base_model_sha256"
  cache_file "$host" "$zeva_checkpoint/model.safetensors" \
    "$zeva_cache/model.safetensors" "$zeva_model_sha256"
  cache_file "$host" "$zeva_checkpoint/zeva_adapter.pth" \
    "$zeva_cache/zeva_adapter.pth" "$zeva_adapter_sha256"
done

base_config=$eval_root/configs/base-step-${base_step}.yml
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

validation_plan=$eval_root/validation_plan.json
python3 - "$validation_plan" "$selection" "$base_checkpoint" "$zeva_checkpoint" \
  "$base_step" "$zeva_step" "$base_model_sha256" "$zeva_model_sha256" \
  "$zeva_adapter_sha256" "$episodes" "$base_config" "$zeva_config" \
  "$base_cache" "$zeva_cache" "$model_host_c" "$model_host_d" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(destination, selection, base_checkpoint, zeva_checkpoint, base_step, zeva_step,
 base_hash, zeva_hash, adapter_hash, episodes, base_config, zeva_config,
 base_cache, zeva_cache, model_host_c, model_host_d) = sys.argv[1:]
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "schema": "zeva-robotwin-anchored-v9-closed-loop-validation-plan-v1",
    "selection": str(Path(selection).resolve()),
    "selection_data": "train95_validation5_only",
    "closed_loop_metrics_used_for_checkpoint_selection": False,
    "base_checkpoint_step": int(base_step),
    "zeva_checkpoint_step": int(zeva_step),
    "base_checkpoint": str(Path(base_checkpoint).resolve()),
    "zeva_checkpoint": str(Path(zeva_checkpoint).resolve()),
    "base_model_sha256": base_hash,
    "zeva_model_sha256": zeva_hash,
    "zeva_adapter_sha256": adapter_hash,
    "base_config": str(Path(base_config).resolve()),
    "base_config_sha256": digest(base_config),
    "zeva_config": str(Path(zeva_config).resolve()),
    "zeva_config_sha256": digest(zeva_config),
    "runtime_cache": {
        "base_checkpoint": base_cache,
        "zeva_checkpoint": zeva_cache,
        "replicated_model_hosts": [model_host_c, model_host_d],
        "identity": {
            "base_model_sha256": base_hash,
            "zeva_model_sha256": zeva_hash,
            "zeva_adapter_sha256": adapter_hash,
        },
    },
    "episodes_per_task_per_split": int(episodes),
    "splits": {"c": {"start_seed": 7000}, "d": {"start_seed": 8000}},
    "excluded_prior_validation_starts": [5000, 6000],
    "reserved_final_start_seed": 10000,
    "acceptance": "delta>=0 on each split and combined ZeVA-Base gain>=6/160",
}
path = Path(destination)
if path.is_file() and json.loads(path.read_text()) != payload:
    raise RuntimeError("existing anchored-v9 validation plan differs; refusing mutation")
if not path.is_file():
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
PY

run_split() {
  local name=$1 start_seed=$2 model_host=$3 model_ip=$4 render_host=$5 render_runtime=$6
  local render_mps_pipe_directory=${7:-}
  local output=$eval_root/$name
  if [[ -s "$output/paired_report.json" ]]; then
    return
  fi
  env MODEL_HOST="$model_host" MODEL_IP="$model_ip" \
    RENDER_HOST="$render_host" RENDER_RUNTIME="$render_runtime" \
    RENDER_MPS_PIPE_DIRECTORY="$render_mps_pipe_directory" \
    OUTPUT_ROOT="$output" BASELINE_CONFIG="$base_config" ZEVA_CONFIG="$zeva_config" \
    ANCHOR_CONFIG="" BASELINE_IS_UNTOUCHED_ANCHOR=false \
    REQUIRE_EXPLICIT_FOUNDATION=true FOUNDATION_MODEL_SHA256="$foundation_sha256" \
    TASK_MANIFEST="$task_manifest" EPISODES="$episodes" ABSOLUTE_START_SEED="$start_seed" \
    MIN_BASELINE_SUCCESS_RATE=0 \
    BASELINE_LABEL="anchored-v9-base-step-$base_step-$name" \
    ZEVA_LABEL="anchored-v9-zeva-step-$zeva_step-$name" \
    bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh" \
    > "$eval_root/$name.launcher.log" 2>&1
}

run_split split-c 7000 "$model_host_c" "$model_ip_c" "$render_host_c" "$render_runtime_c" "" &
pid_c=$!
run_split split-d 8000 "$model_host_d" "$model_ip_d" "$render_host_d" "$render_runtime_d" \
  /tmp/zeva-v9-split-d-mps-bypass &
pid_d=$!
printf '%s\n' "$pid_c" > "$eval_root/split-c.pid"
printf '%s\n' "$pid_d" > "$eval_root/split-d.pid"
status=0
wait "$pid_c" || status=1
wait "$pid_d" || status=1
(( status == 0 )) || exit "$status"

python3 - "$eval_root" "$episodes" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
episodes = int(sys.argv[2])
reports = {name: json.loads((root / name / "paired_report.json").read_text())
           for name in ("split-c", "split-d")}
manifests = {name: json.loads((root / name / "seed_manifest.json").read_text())
             for name in reports}
for name, report in reports.items():
    if report.get("total_paired_episodes") != episodes * 10:
        raise RuntimeError(f"{name}: incomplete paired report")
seed_sets = {
    name: {task: {int(row["seed"]) for row in rows}
           for task, rows in payload["tasks"].items()}
    for name, payload in manifests.items()
}
overlap = {task: sorted(seed_sets["split-c"][task] & seed_sets["split-d"][task])
           for task in seed_sets["split-c"]}
if any(overlap.values()):
    raise RuntimeError(f"anchored-v9 validation split overlap: {overlap}")
rows = []
for name, report in reports.items():
    count = round(report["absolute_delta"] * report["total_paired_episodes"])
    rows.append({
        "split": name,
        "baseline_success_rate": report["baseline_success_rate"],
        "zeva_success_rate": report["zeva_success_rate"],
        "delta_count": count,
        "episodes_per_condition": report["total_paired_episodes"],
    })
combined_delta = sum(row["delta_count"] for row in rows)
accepted = all(row["delta_count"] >= 0 for row in rows) and combined_delta >= 6
payload = {
    "schema": "zeva-robotwin-anchored-v9-closed-loop-validation-summary-v1",
    "accepted_for_fresh_final_test": accepted,
    "criteria": {"each_split_delta_nonnegative": True,
                 "minimum_combined_delta_count": 6},
    "splits_pairwise_disjoint": True,
    "rows": rows,
    "combined_delta_count": combined_delta,
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
