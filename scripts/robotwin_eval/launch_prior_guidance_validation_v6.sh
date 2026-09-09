#!/usr/bin/env bash
set -euo pipefail

# Validate a BehaviorVLA-strength Gaussian action-prior branch without touching
# PI0.5 or consuming the formal seed-1000 test set.  Split A and split B run on
# independent model/render host pairs and reuse their immutable paired Base
# evidence.  The context projector is exactly zero in this diagnostic.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-prior-guidance-v6}
source_checkpoint=${SOURCE_CHECKPOINT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-safe-router-v3/zeva/001250}
context_gate_probability=${CONTEXT_GATE_PROBABILITY:-0.0}
prior_gate_probability=${PRIOR_GATE_PROBABILITY:-0.5}
candidate_tag=${CANDIDATE_TAG:-context0-prior05}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}

split_a_source=${SPLIT_A_SOURCE:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v4-closed-loop/validation-scale-025}
split_b_source=${SPLIT_B_SOURCE:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v5-multisplit/validation-scale-025}
split_a_expected=${SPLIT_A_EXPECTED_SUCCESSES:-50}
split_b_expected=${SPLIT_B_EXPECTED_SUCCESSES:-43}

model_host_a=${MODEL_HOST_A:-aigc29}
model_ip_a=${MODEL_IP_A:-172.16.80.163}
render_host_a=${RENDER_HOST_A:-aigc24}
render_runtime_a=${RENDER_RUNTIME_A:-/data1/dingxin/robotwin-formal-eval/RoboTwin}
model_host_b=${MODEL_HOST_B:-aigc29}
model_ip_b=${MODEL_IP_B:-172.16.80.163}
render_host_b=${RENDER_HOST_B:-aigc24}
render_runtime_b=${RENDER_RUNTIME_B:-/data1/dingxin/robotwin-formal-eval/RoboTwin}

candidate_dir=$run_root/candidates/$candidate_tag
candidate_adapter=$candidate_dir/zeva_adapter.pth
candidate_config=$run_root/configs/$candidate_tag.yml
mkdir -p "$candidate_dir" "$run_root/configs"

test -s "$source_checkpoint/model.safetensors"
test -s "$source_checkpoint/zeva_adapter.pth"
test -s "$split_a_source/baseline/report.json"
test -s "$split_b_source/baseline/report.json"
test -s "$split_a_source/seed_manifest.json"
test -s "$split_b_source/seed_manifest.json"

python3 "$zeva_root/scripts/make_robotwin_branch_guidance_candidate.py" \
  --adapter "$source_checkpoint/zeva_adapter.pth" \
  --output "$candidate_adapter" \
  --context-gate-probability "$context_gate_probability" \
  --prior-gate-probability "$prior_gate_probability" \
  --candidate-set "dual-disjoint-split-prior-guidance-v6"
ln -sfn "$source_checkpoint/model.safetensors" "$candidate_dir/model.safetensors"

python3 - "$candidate_config" "$candidate_dir" <<'PY'
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

run_split() {
  local name=$1
  local source=$2
  local expected=$3
  local model_host=$4
  local model_ip=$5
  local render_host=$6
  local render_runtime=$7
  local output=$run_root/$candidate_tag/$name
  if [[ -s "$output/paired_report.json" ]]; then
    return
  fi
  env MODEL_HOST="$model_host" MODEL_IP="$model_ip" RENDER_HOST="$render_host" \
    RENDER_RUNTIME="$render_runtime" OUTPUT_ROOT="$output" \
    BASELINE_CONFIG="$zeva_root/scripts/robotwin_eval/baseline_model_config.yml" \
    ZEVA_CONFIG="$candidate_config" ANCHOR_CONFIG="" TASK_MANIFEST="$task_manifest" \
    EPISODES=8 FROZEN_SEED_MANIFEST="$source/seed_manifest.json" \
    PRECOMPUTED_BASELINE_ROOT="$source/baseline" \
    PRECOMPUTED_BASELINE_EXPECTED_SUCCESSES="$expected" \
    BASELINE_LABEL="immutable-validation-$name-pi05" \
    ZEVA_LABEL="$candidate_tag-validation-$name" \
    bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh" \
    > "$run_root/$candidate_tag-$name.launcher.log" 2>&1
}

printf '{"state":"running_validation","candidate":"%s","started":"%s"}\n' \
  "$candidate_tag" "$(date -Iseconds)" > "$run_root/state.json"
status=0
if [[ "$model_host_a" == "$model_host_b" || "$render_host_a" == "$render_host_b" ]]; then
  # Sharing either host would make parallel runs contend for the same eight
  # GPUs/ports.  The default uses the one fully validated Mamba runtime and
  # therefore runs the disjoint splits sequentially.
  run_split split-a "$split_a_source" "$split_a_expected" \
    "$model_host_a" "$model_ip_a" "$render_host_a" "$render_runtime_a" || status=1
  if (( status == 0 )); then
    run_split split-b "$split_b_source" "$split_b_expected" \
      "$model_host_b" "$model_ip_b" "$render_host_b" "$render_runtime_b" || status=1
  fi
else
  run_split split-a "$split_a_source" "$split_a_expected" \
    "$model_host_a" "$model_ip_a" "$render_host_a" "$render_runtime_a" &
  pid_a=$!
  run_split split-b "$split_b_source" "$split_b_expected" \
    "$model_host_b" "$model_ip_b" "$render_host_b" "$render_runtime_b" &
  pid_b=$!
  printf '%s\n' "$pid_a" > "$run_root/$candidate_tag-split-a.pid"
  printf '%s\n' "$pid_b" > "$run_root/$candidate_tag-split-b.pid"
  wait "$pid_a" || status=1
  wait "$pid_b" || status=1
fi
if (( status != 0 )); then
  printf '{"state":"validation_failed","candidate":"%s","finished":"%s"}\n' \
    "$candidate_tag" "$(date -Iseconds)" > "$run_root/state.json"
  exit "$status"
fi

python3 - "$run_root" "$candidate_tag" <<'PY'
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
candidate = sys.argv[2]
rows = {}
for split in ("split-a", "split-b"):
    report = json.loads((root / candidate / split / "paired_report.json").read_text())
    rows[split] = {
        "baseline_successes": round(report["baseline_success_rate"] * 80),
        "zeva_successes": round(report["zeva_success_rate"] * 80),
        "gain": round((report["zeva_success_rate"] - report["baseline_success_rate"]) * 80),
        "paired_report": str(root / candidate / split / "paired_report.json"),
    }
payload = {
    "schema": "zeva-robotwin-prior-guidance-validation-v1",
    "candidate": candidate,
    "splits": rows,
    "non_negative_each_split": all(row["gain"] >= 0 for row in rows.values()),
    "aggregate_gain": sum(row["gain"] for row in rows.values()),
    "formal_test_metrics_used": False,
}
temporary = root / "validation_summary.json.partial"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "validation_summary.json")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
printf '{"state":"validation_complete","candidate":"%s","finished":"%s"}\n' \
  "$candidate_tag" "$(date -Iseconds)" > "$run_root/state.json"
