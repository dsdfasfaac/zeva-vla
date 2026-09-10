#!/usr/bin/env bash
set -euo pipefail

# Mechanism-level paired closed-loop validation for one offline-selected v11
# checkpoint. The two splits use disjoint environment seeds and different
# continuous diffusion RNG streams. Neither split may be used for final
# seed-1000 reporting.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
train_root=${TRAIN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-prior-action-expert-v11/zeva}
step=${ZEVA_STEP:-500}
checkpoint=$train_root/$(printf '%06d' "$step")
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-prior-action-expert-v11/step-$(printf '%06d' "$step")}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
foundation=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
foundation_sha256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
episodes=${EPISODES:-8}
split_i_seed=${SPLIT_I_SEED:-15000}
split_j_seed=${SPLIT_J_SEED:-16000}
split_i_rng=${SPLIT_I_RNG:-20260907}
split_j_rng=${SPLIT_J_RNG:-20260908}
model_cache_root=${MODEL_CACHE_ROOT:-/data1/dingxin/zeva-eval-cache/prior-action-expert-v11}
render_vulkan_icd=${RENDER_VULKAN_ICD:-/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin/.venv_robotwin/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json}
render_ld_library_path=${RENDER_LD_LIBRARY_PATH:-/data2/dingxin:/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/lib}

test -s "$checkpoint/model.safetensors"
test -s "$checkpoint/zeva_adapter.pth"
test -s "$checkpoint/checkpoint_audit.json"
test -s "$checkpoint/residual_branch_audit.json"
test -s "$task_manifest"
test "$(sha256sum "$foundation/model.safetensors" | awk '{print $1}')" = "$foundation_sha256"
python3 - "$checkpoint/checkpoint_audit.json" "$checkpoint/residual_branch_audit.json" <<'PY'
import json, sys
checkpoint, residual = (json.load(open(path)) for path in sys.argv[1:])
if checkpoint.get("passed") is not True:
    raise SystemExit("v11 checkpoint identity/invariant audit did not pass")
if checkpoint["foundation_drift"]["frozen_changed_count"] != 0:
    raise SystemExit("v11 changed a frozen foundation tensor")
aggregate = residual["aggregate"]
if aggregate["prior_injection_horizon"] != 15:
    raise SystemExit("v11 prior is not H15")
if aggregate["injected_context_residual_rms"] != 0:
    raise SystemExit("v11 direct context residual is nonzero")
if not 0.499999 <= aggregate["prior_gate"]["minimum"] <= aggregate["prior_gate"]["maximum"] <= 0.500001:
    raise SystemExit("v11 prior guidance is not fixed at 0.5")
PY

candidate_sha256=$(sha256sum "$checkpoint/model.safetensors" | awk '{print $1}')
adapter_sha256=$(sha256sum "$checkpoint/zeva_adapter.pth" | awk '{print $1}')
mkdir -p "$eval_root/configs"

cache_file() {
  local host=$1 source=$2 destination=$3 expected=$4
  ssh "$host" "set -euo pipefail
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

prepare_host() {
  local host=$1
  local base_cache=$model_cache_root/base-$foundation_sha256
  local zeva_cache=$model_cache_root/step-${step}-${candidate_sha256}
  cache_file "$host" "$foundation/model.safetensors" "$base_cache/model.safetensors" "$foundation_sha256"
  cache_file "$host" "$checkpoint/model.safetensors" "$zeva_cache/model.safetensors" "$candidate_sha256"
  cache_file "$host" "$checkpoint/zeva_adapter.pth" "$zeva_cache/zeva_adapter.pth" "$adapter_sha256"
}

prepare_host aigc24 &
cache_i=$!
prepare_host aigc28 &
cache_j=$!
wait "$cache_i"
wait "$cache_j"

python3 - "$eval_root/configs" "$model_cache_root" "$step" "$candidate_sha256" \
  "$foundation_sha256" "$split_i_rng" "$split_j_rng" <<'PY'
import os, sys
from pathlib import Path
root, cache_root, step, candidate_hash, foundation_hash, rng_i, rng_j = sys.argv[1:]
root = Path(root)
shared = """policy_name: zeva_policy
handoff_root: /mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
foundation_checkpoint: /mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1/checkpoint/pretrained_model-best-v1
"""
for split, rng in (("i", rng_i), ("j", rng_j)):
    base = shared + f"""stage2_checkpoint: {cache_root}/base-{foundation_hash}
baseline_only: true
device: cuda
model_rng_seed: {rng}
"""
    zeva = shared + f"""goal_embedding_checkpoint: /mnt/100T/users/dingxin/VLA/runtime/pretrained_model-stage1-language-v1
stage2_checkpoint: {cache_root}/step-{step}-{candidate_hash}
zte_checkpoint: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/zte_best.pth
causal_bank: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/train_causal_bank.pt
retrieval_checkpoint: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1.5-task-retrieval/task_retrieval.pth
baseline_only: false
device: cuda
model_rng_seed: {rng}
"""
    for name, value in ((f"base-{split}.yml", base), (f"zeva-{split}.yml", zeva)):
        destination = root / name
        if destination.is_file() and destination.read_text() != value:
            raise RuntimeError(f"refusing to mutate {destination}")
        if not destination.is_file():
            temporary = destination.with_suffix(destination.suffix + ".partial")
            temporary.write_text(value)
            os.replace(temporary, destination)
PY

run_split() {
  local name=$1 start_seed=$2 rng=$3 model_host=$4 model_ip=$5 render_host=$6 render_runtime=$7
  local output=$eval_root/split-$name
  if [[ -s "$output/paired_report.json" ]]; then return; fi
  env MODEL_HOST="$model_host" MODEL_IP="$model_ip" RENDER_HOST="$render_host" \
    RENDER_RUNTIME="$render_runtime" RENDER_MPS_PIPE_DIRECTORY="/tmp/zeva-v11-step-${step}-${name}-mps" \
    RENDER_VULKAN_ICD="$render_vulkan_icd" \
    RENDER_LD_LIBRARY_PATH="$render_ld_library_path" \
    OUTPUT_ROOT="$output" BASELINE_CONFIG="$eval_root/configs/base-${name}.yml" \
    ZEVA_CONFIG="$eval_root/configs/zeva-${name}.yml" ANCHOR_CONFIG="" \
    BASELINE_IS_UNTOUCHED_ANCHOR=true REQUIRE_EXPLICIT_FOUNDATION=true \
    FOUNDATION_MODEL_SHA256="$foundation_sha256" TASK_MANIFEST="$task_manifest" \
    EPISODES="$episodes" ABSOLUTE_START_SEED="$start_seed" MODEL_RNG_SEED="$rng" \
    MODEL_SEED_POLICY=continuous MIN_BASELINE_SUCCESS_RATE=0 \
    BASELINE_LABEL="untouched-best-v1-split-${name}" ZEVA_LABEL="v11-step-${step}-split-${name}" \
    bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh" \
    > "$eval_root/split-${name}.launcher.log" 2>&1
}

run_split i "$split_i_seed" "$split_i_rng" aigc24 172.16.80.158 aigc29 /mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin &
pid_i=$!
run_split j "$split_j_seed" "$split_j_rng" aigc28 172.16.80.162 aigc28 /mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin &
pid_j=$!
printf '%s\n' "$pid_i" > "$eval_root/split-i.pid"
printf '%s\n' "$pid_j" > "$eval_root/split-j.pid"
status=0
wait "$pid_i" || status=1
wait "$pid_j" || status=1
(( status == 0 )) || exit "$status"

python3 "$zeva_root/scripts/robotwin_eval/audit_advantage10_formal.py" "$eval_root/split-i"
python3 "$zeva_root/scripts/robotwin_eval/audit_advantage10_formal.py" "$eval_root/split-j"
python3 - "$eval_root" "$episodes" <<'PY'
import json, sys
from pathlib import Path
root, episodes = Path(sys.argv[1]), int(sys.argv[2])
rows=[]
for name in ("i", "j"):
    report=json.loads((root/f"split-{name}"/"paired_report.json").read_text())
    if report["total_paired_episodes"] != episodes*10:
        raise RuntimeError(f"split-{name} is incomplete")
    rows.append({"split":name,"baseline_success_rate":report["baseline_success_rate"],
                 "zeva_success_rate":report["zeva_success_rate"],
                 "delta_count":round(report["absolute_delta"]*report["total_paired_episodes"]),
                 "episodes_per_condition":report["total_paired_episodes"]})
combined=sum(row["delta_count"] for row in rows)
accepted=all(row["delta_count"]>=0 for row in rows) and combined>=6
payload={"schema":"zeva-robotwin-v11-closed-loop-validation-v1",
         "accepted_for_independent_confirmation":accepted,"rows":rows,
         "combined_delta_count":combined,"combined_episodes_per_condition":sum(row["episodes_per_condition"] for row in rows),
         "criteria":"each split delta>=0 and combined gain>=6/160"}
(root/"validation_summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2))
if not accepted: raise SystemExit(4)
PY
