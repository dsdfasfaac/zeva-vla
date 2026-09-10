#!/usr/bin/env bash
set -euo pipefail

# Formal paired RoboTwin evaluation.  The baseline first selects the first 20
# expert-valid seeds per task from ABSOLUTE_START_SEED (1000 by default); ZeVA
# then reuses that exact sparse
# seed list. Policy recurrent state resets per episode while diffusion RNG
# remains continuous, matching the frozen PI0.5 evaluation handoff.

model_host=${MODEL_HOST:-aigc29}
render_host=${RENDER_HOST:-aigc24}
model_ip=${MODEL_IP:-172.16.80.163}
slots=${SLOTS:-8}
episodes=${EPISODES:-20}
absolute_start_seed=${ABSOLUTE_START_SEED:-1000}
model_seed_policy=${MODEL_SEED_POLICY:-continuous}
model_rng_seed=${MODEL_RNG_SEED:-20260907}
# Optional previously established normal-PI floor.  A newly evaluated anchor
# can fluctuate slightly because GPU PhysX is not bit deterministic, so the
# formal gate must not silently weaken a historical reference supplied by the
# experiment protocol.
min_baseline_success_rate=${MIN_BASELINE_SUCCESS_RATE:-0}
baseline_is_untouched_anchor=${BASELINE_IS_UNTOUCHED_ANCHOR:-false}
require_explicit_foundation=${REQUIRE_EXPLICIT_FOUNDATION:-false}
foundation_model_sha256=${FOUNDATION_MODEL_SHA256:-}
precomputed_baseline_root=${PRECOMPUTED_BASELINE_ROOT:-}
# Formal 10x20 imports retain the historical 114-success default.  Calibration
# launchers must pass their independently observed validation count explicitly.
precomputed_baseline_expected_successes=${PRECOMPUTED_BASELINE_EXPECTED_SUCCESSES:-114}
base_port=${BASE_PORT:-19200}
zeva_root=/mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA
shared_runtime=${SHARED_RUNTIME:-/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin}
# The validated a24 deployment keeps a node-local copy for I/O speed.  Other
# idle render nodes may use the identical shared checkout/venv when evaluating
# checkpoint candidates in parallel; making the path explicit prevents a
# silent fallback to a different RoboTwin installation.
render_runtime=${RENDER_RUNTIME:-/data1/dingxin/robotwin-formal-eval/RoboTwin}
render_vulkan_icd=${RENDER_VULKAN_ICD:-/usr/share/vulkan/icd.d/nvidia_icd.json}
render_mps_pipe_directory=${RENDER_MPS_PIPE_DIRECTORY:-}
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
release_runtime=$handoff/runtime
native_transformers=/data1/dingxin/transformers5-runtime
shared_py310_deps=/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1
output=${OUTPUT_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-paired-stage2a-h15-seen-v1}
baseline_config=${BASELINE_CONFIG:-$zeva_root/scripts/robotwin_eval/baseline_model_config.yml}
zeva_config=${ZEVA_CONFIG:-$zeva_root/scripts/robotwin_eval/model_config.yml}
anchor_config=${ANCHOR_CONFIG:-}
task_manifest=${TASK_MANIFEST:-}
baseline_label=${BASELINE_LABEL:-pi05-baseline-h15}
zeva_label=${ZEVA_LABEL:-stage2a-005000-h15}
anchor_label=${ANCHOR_LABEL:-pi05-anchor-h15}
seed_manifest=$output/seed_manifest.json
frozen_seed_manifest=${FROZEN_SEED_MANIFEST:-}
baseline_only_precompute=${BASELINE_ONLY_PRECOMPUTE:-false}

if [[ -n "$precomputed_baseline_root" ]]; then
  baseline_is_untouched_anchor=true
  if [[ -z "$frozen_seed_manifest" ]]; then
    echo "PRECOMPUTED_BASELINE_ROOT requires FROZEN_SEED_MANIFEST" >&2
    exit 2
  fi
fi
if ! [[ "$precomputed_baseline_expected_successes" =~ ^[0-9]+$ ]]; then
  echo "PRECOMPUTED_BASELINE_EXPECTED_SUCCESSES must be a non-negative integer" >&2
  exit 2
fi
if [[ "$baseline_is_untouched_anchor" != true && "$baseline_is_untouched_anchor" != false ]]; then
  echo "BASELINE_IS_UNTOUCHED_ANCHOR must be true or false" >&2
  exit 2
fi
if [[ "$require_explicit_foundation" != true && "$require_explicit_foundation" != false ]]; then
  echo "REQUIRE_EXPLICIT_FOUNDATION must be true or false" >&2
  exit 2
fi
if [[ "$baseline_only_precompute" != true && "$baseline_only_precompute" != false ]]; then
  echo "BASELINE_ONLY_PRECOMPUTE must be true or false" >&2
  exit 2
fi
if [[ "$require_explicit_foundation" == true && ! "$foundation_model_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "explicit-foundation runs require a 64-character FOUNDATION_MODEL_SHA256" >&2
  exit 2
fi
test -s "$baseline_config"
test -s "$zeva_config"
baseline_config_sha256=$(sha256sum "$baseline_config" | awk '{print $1}')
zeva_config_sha256=$(sha256sum "$zeva_config" | awk '{print $1}')
anchor_config_sha256=""
if [[ -n "$anchor_config" ]]; then
  test -s "$anchor_config"
  anchor_config_sha256=$(sha256sum "$anchor_config" | awk '{print $1}')
fi

mkdir -p "$output"
render_mps_env=""
if [[ -n "$render_mps_pipe_directory" ]]; then
  ssh "$render_host" "mkdir -p '$render_mps_pipe_directory'"
  render_mps_env="CUDA_MPS_PIPE_DIRECTORY='$render_mps_pipe_directory'"
fi

seed_selection="baseline-selects-first-${episodes}-expert-valid-from-${absolute_start_seed};anchor-and-zeva-replay-exact-list"
seed_manifest_source=""
seed_manifest_source_sha256=""
precomputed_baseline_report_sha256=""
if [[ -n "$frozen_seed_manifest" ]]; then
  cp "$frozen_seed_manifest" "$seed_manifest"
  seed_selection="all-conditions-replay-existing-frozen-expert-valid-manifest"
  seed_manifest_source=$frozen_seed_manifest
  seed_manifest_source_sha256=$(sha256sum "$frozen_seed_manifest" | awk '{print $1}')
fi

if [[ -n "$precomputed_baseline_root" ]]; then
  precomputed_baseline_report_sha256=$(sha256sum "$precomputed_baseline_root/report.json" | awk '{print $1}')
fi

python3 - "$shared_runtime/task_config/_eval_step_limit.yml" "$task_manifest" "$output/tasks.txt" <<'PY'
import json
import sys
import yaml

source, manifest, destination = sys.argv[1:]
all_tasks = list(yaml.safe_load(open(source, encoding="utf-8")))
if len(all_tasks) != 50:
    raise SystemExit(f"expected 50 RoboTwin tasks in runtime, got {len(all_tasks)}")
if manifest:
    payload = json.load(open(manifest, encoding="utf-8"))
    tasks = list(payload["task_names"])
    missing = sorted(set(tasks) - set(all_tasks))
    if missing:
        raise SystemExit(f"task subset contains unknown RoboTwin tasks: {missing}")
    if len(tasks) != len(set(tasks)):
        raise SystemExit("task subset contains duplicate tasks")
else:
    tasks = all_tasks
open(destination, "w", encoding="utf-8").write("\n".join(tasks) + "\n")
PY
mapfile -t tasks < "$output/tasks.txt"
task_count=${#tasks[@]}
expected_total_episodes=$((task_count * episodes))

model_pythonpath=$(ssh "$model_host" "python3 - <<'PY'
import site
print(site.getsitepackages()[0])
PY
")
# Keep the checkpoint's native Transformers 5 Python package ahead of the
# Python-3.10 dependency overlay.  Only pure-Python packages and distribution
# metadata are linked here; compiled dependencies continue to come from the
# host's tested 3.10 runtime.  A stock best-v1 anchor must pass before results
# from this launcher are reportable.
ssh "$model_host" "mkdir -p '$native_transformers'; \
  ln -sfn '$release_runtime/lerobot-main-deps-py311-v1/transformers' '$native_transformers/transformers'; \
  ln -sfn '$release_runtime/lerobot-main-deps-py311-v1/transformers-5.5.4.dist-info' '$native_transformers/transformers-5.5.4.dist-info'; \
  ln -sfn '$release_runtime/lerobot-main-deps-py311-v1/huggingface_hub' '$native_transformers/huggingface_hub'; \
  ln -sfn '$release_runtime/lerobot-main-deps-py311-v1/huggingface_hub-1.27.0.dist-info' '$native_transformers/huggingface_hub-1.27.0.dist-info'; \
  ln -sfn '$release_runtime/lerobot-main-deps-py311-v1/tokenizers-0.22.2.dist-info' '$native_transformers/tokenizers-0.22.2.dist-info'"
# Put the experiment's adapter before RoboTwin's bundled policy directory.
# Otherwise policy_model_server.py appends ./policy and can silently import a
# stale ZeVA adapter that does not load the Stage 2 foundation weights.
model_pythonpath=$zeva_root/scripts/robotwin_eval:$native_transformers:/data1/dingxin/zeva-runtime-deps:$shared_py310_deps:$model_pythonpath:$release_runtime/h100-extra-deps:$release_runtime/lerobot-overlay-v2:$release_runtime/lerobot-main-py311-v1/src:$release_runtime/src:$zeva_root/src:$zeva_root:$release_runtime/lerobot-main-deps-py311-v1

server_pids=()
stop_servers() {
  for pid in "${server_pids[@]:-}"; do
    [[ -n "$pid" ]] && ssh "$model_host" "kill '$pid' 2>/dev/null || true" || true
  done
  server_pids=()
}
trap stop_servers EXIT INT TERM

start_servers() {
  local config=$1
  local condition_root=$2
  mkdir -p "$condition_root/logs" "$condition_root/server-pids"
  server_pids=()
  for slot in $(seq 0 $((slots - 1))); do
    local port=$((base_port + slot))
    local log="$condition_root/logs/server-slot${slot}.log"
    local pid
    pid=$(ssh "$model_host" "cd '$shared_runtime'; nohup env \
      PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='$slot' PYTHONPATH='$model_pythonpath' \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
      python3 script/policy_model_server.py --port '$port' --config '$config' \
        --overrides --host 0.0.0.0 > '$log' 2>&1 < /dev/null & echo \$!")
    printf '%s\n' "$pid" > "$condition_root/server-pids/slot${slot}.pid"
    server_pids+=("$pid")
  done
  for slot in $(seq 0 $((slots - 1))); do
    local port=$((base_port + slot))
    for _ in $(seq 1 180); do
      ssh "$model_host" "ss -ltn | grep -q ':$port '" && break
      sleep 5
    done
    ssh "$model_host" "ss -ltn | grep -q ':$port '" || {
      echo "slot $slot model server did not become ready" >&2
      return 1
    }
  done
}

run_condition() {
  local condition=$1
  local config=$2
  local ckpt_label=$3
  local seeds=${4:-}
  local condition_root="$output/$condition"
  mkdir -p "$condition_root"/{logs,progress,status,results}
  printf '{"state":"starting","condition":"%s","started":"%s"}\n' \
    "$condition" "$(date -Iseconds)" > "$condition_root/state.json"
  start_servers "$config" "$condition_root"

  local workers=()
  for slot in $(seq 0 $((slots - 1))); do
    (
      local port=$((base_port + slot))
      for index in "${!tasks[@]}"; do
        (( index % slots == slot )) || continue
        local task=${tasks[$index]}
        local progress="$condition_root/progress/$task.json"
        local log="$condition_root/logs/$task.log"
        local status="$condition_root/status/$task.json"
        local result_dir="$condition_root/results/$task"
        local seed_args="--fixed_seed_sequence False"
        if [[ -n "$seeds" ]]; then
          seed_args="--fixed_seed_sequence True --seed_manifest '$seeds'"
        fi
        local started
        started=$(date -Iseconds)
        set +e
        ssh "$render_host" "cd '$render_runtime'; env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='$slot' VK_ICD_FILENAMES='$render_vulkan_icd' $render_mps_env PYTHONPATH='$zeva_root/scripts/robotwin_eval:$render_runtime/script:$render_runtime:$render_runtime/policy' \
          .venv_robotwin/bin/python '$zeva_root/scripts/robotwin_eval/eval_policy_client.py' --port '$port' --config '$zeva_root/scripts/robotwin_eval/client_config.yml' \
          --overrides --task_name '$task' --task_config zeva_randomized --test_num '$episodes' \
          --instruction_type seen --seed 0 --absolute_start_seed '$absolute_start_seed' $seed_args \
          --model_seed_policy '$model_seed_policy' \
          --policy_name zeva_policy --ckpt_setting '$ckpt_label' \
          --eval_video_log True --result_dir '$result_dir' \
          --execute_horizon 15 --chunk_length 50 --action_dim 16 \
          --server_host '$model_ip' --resume_progress_path '$progress'" > "$log" 2>&1
        local rc=$?
        set -e
        printf '{"job":"%s","slot":%d,"return_code":%d,"started":"%s","finished":"%s"}\n' \
          "$task" "$slot" "$rc" "$started" "$(date -Iseconds)" > "$status"
        (( rc == 0 )) || echo "$condition/$task failed with rc=$rc; progress is resumable" >&2
      done
    ) > "$condition_root/logs/worker-slot${slot}.log" 2>&1 &
    workers+=("$!")
  done
  printf '%s\n' "${workers[@]}" > "$condition_root/worker-pids.txt"
  printf '{"state":"running","condition":"%s","started":"%s","jobs":%d}\n' \
    "$condition" "$(date -Iseconds)" "${#tasks[@]}" > "$condition_root/state.json"

  local worker_rc=0
  for pid in "${workers[@]}"; do
    wait "$pid" || worker_rc=1
  done
  stop_servers

  python3 - "$condition_root" "$episodes" "$condition" "$model_seed_policy" "$absolute_start_seed" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
condition = sys.argv[3]
model_seed_policy = sys.argv[4]
absolute_start_seed = int(sys.argv[5])
rows = []
for task in (root.parent / "tasks.txt").read_text().splitlines():
    progress_path = root / "progress" / f"{task}.json"
    status_path = root / "status" / f"{task}.json"
    if not progress_path.is_file():
        raise RuntimeError(f"{condition}/{task}: missing progress")
    if not status_path.is_file():
        raise RuntimeError(f"{condition}/{task}: missing terminal status")
    progress = json.loads(progress_path.read_text())
    status = json.loads(status_path.read_text())
    if status.get("return_code") != 0:
        raise RuntimeError(f"{condition}/{task}: nonzero terminal status {status}")
    episodes = progress["episode_results"]
    seeds = [int(item["seed"]) for item in episodes]
    if not progress.get("complete") or len(episodes) != expected:
        raise RuntimeError(f"{condition}/{task}: incomplete progress")
    if any(seed < absolute_start_seed for seed in seeds) or len(set(seeds)) != expected:
        raise RuntimeError(f"{condition}/{task}: invalid seeds {seeds}")
    videos = sorted((root / "results" / task).glob("episode*_randomized-true_success-*.mp4"))
    if len(videos) != expected:
        raise RuntimeError(f"{condition}/{task}: expected {expected} videos, got {len(videos)}")
    successes = sum(bool(item["success"]) for item in episodes)
    video_successes = sum("success-true" in video.name for video in videos)
    if successes != video_successes:
        raise RuntimeError(f"{condition}/{task}: progress/video success mismatch")
    rows.append({"task": task, "episodes": expected, "successes": successes,
                 "success_rate": successes / expected, "seeds": seeds})
report = {
    "schema": "zeva-robotwin-paired-condition-v1",
    "condition": condition,
    "profile": "randomized_seen_large_d435",
    "action_contract": "eef16_h50_execute_h15",
    "model_seed_policy": model_seed_policy,
    "task_count": len(rows),
    "episodes_per_task": expected,
    "total_episodes": len(rows) * expected,
    "total_successes": sum(row["successes"] for row in rows),
    "micro_success_rate": sum(row["successes"] for row in rows) / (len(rows) * expected),
    "macro_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
    "tasks": rows,
}
temporary = root / "report.json.partial"
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "report.json")
PY
  printf '{"state":"complete","condition":"%s","finished":"%s","jobs":%d}\n' \
    "$condition" "$(date -Iseconds)" "${#tasks[@]}" > "$condition_root/state.json"
  # A task may have been resumed by an audited recovery process after its
  # original worker exited.  In that case the historical wait status is stale;
  # the complete progress/video/terminal-status audit above is authoritative.
  if (( worker_rc != 0 )); then
    echo "$condition: recovered worker failure; final condition audit passed" >&2
  fi
  return 0
}

cat > "$output/manifest.json" <<EOF
{
  "schema": "zeva-robotwin-formal-paired-eval-v1",
  "baseline_config": "$baseline_config",
  "baseline_config_sha256": "$baseline_config_sha256",
  "zeva_config": "$zeva_config",
  "zeva_config_sha256": "$zeva_config_sha256",
  "anchor_config": "$anchor_config",
  "anchor_config_sha256": "$anchor_config_sha256",
  "model_identity_contract": "$(if [[ "$require_explicit_foundation" == true ]]; then printf explicit-foundation-v1; else printf legacy-config-v1; fi)",
  "foundation_model_sha256": "$foundation_model_sha256",
  "task_manifest": "$task_manifest",
  "task_count": $task_count,
  "episodes_per_task": $episodes,
  "absolute_start_seed": $absolute_start_seed,
  "seed_selection": "$seed_selection",
  "seed_manifest_source": "$seed_manifest_source",
  "seed_manifest_source_sha256": "$seed_manifest_source_sha256",
  "instruction_type": "seen",
  "camera": "Large_D435_640x480",
  "action_contract": "chunk-start-relative-eef16-predict-h50-execute-h15",
  "model_seed_policy": "$model_seed_policy",
  "model_rng_seed": $model_rng_seed,
  "min_baseline_success_rate": $min_baseline_success_rate,
  "baseline_is_untouched_anchor": $baseline_is_untouched_anchor,
  "precomputed_baseline_root": "$precomputed_baseline_root",
  "precomputed_baseline_report_sha256": "$precomputed_baseline_report_sha256",
  "precomputed_baseline_expected_successes": $precomputed_baseline_expected_successes,
  "model_runtime": "native_handoff_transformers_5.5.4",
  "model_host": "$model_host",
  "render_host": "$render_host",
  "render_mps_pipe_directory": "$render_mps_pipe_directory",
  "slots": $slots
}
EOF

if [[ -n "$precomputed_baseline_root" ]]; then
  if [[ -e "$output/baseline" ]]; then
    python3 - "$output/baseline/report.json" "$precomputed_baseline_report_sha256" \
      "$expected_total_episodes" "$precomputed_baseline_expected_successes" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
provenance = report.get("provenance", {})
if (report.get("condition") != "baseline"
        or report.get("total_episodes") != int(sys.argv[3])
        or report.get("total_successes") != int(sys.argv[4])
        or provenance.get("source_report_sha256") != sys.argv[2]):
    raise RuntimeError("existing imported baseline does not match requested untouched PI evidence")
PY
  else
    import_tmp="$output/baseline.importing.$$"
    mkdir -p "$import_tmp"
    cp -al "$precomputed_baseline_root"/. "$import_tmp"/
    python3 - "$import_tmp" "$precomputed_baseline_report_sha256" \
      "$expected_total_episodes" "$precomputed_baseline_expected_successes" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_sha = sys.argv[2]
expected_total = int(sys.argv[3])
expected_successes = int(sys.argv[4])
report_path = root / "report.json"
state_path = root / "state.json"
report = json.loads(report_path.read_text())
if (report.get("total_episodes") != expected_total
        or report.get("total_successes") != expected_successes):
    raise RuntimeError(
        "precomputed untouched PI baseline differs from the preregistered expected count"
    )
report["condition"] = "baseline"
report["provenance"] = {
    "role": "untouched_pi_anchor_established_before_adapter_training",
    "source_report_sha256": expected_sha,
}
temporary = report_path.with_name("report.json.partial")
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
os.replace(temporary, report_path)
state = json.loads(state_path.read_text())
if state.get("state") != "complete":
    raise RuntimeError("precomputed baseline condition is not complete")
state["condition"] = "baseline"
temporary = state_path.with_name("state.json.partial")
temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
os.replace(temporary, state_path)
PY
    mv "$import_tmp" "$output/baseline"
  fi
elif [[ -n "$frozen_seed_manifest" ]]; then
  printf '{"state":"running_baseline","started":"%s"}\n' "$(date -Iseconds)" > "$output/state.json"
  run_condition baseline "$baseline_config" "$baseline_label" "$seed_manifest"
else
  printf '{"state":"running_baseline","started":"%s"}\n' "$(date -Iseconds)" > "$output/state.json"
  run_condition baseline "$baseline_config" "$baseline_label"
fi

if [[ -z "$frozen_seed_manifest" ]]; then
python3 - "$output/baseline/progress" "$output/tasks.txt" "$episodes" "$seed_manifest" "$absolute_start_seed" <<'PY'
import json
import os
import sys
from pathlib import Path

progress_root, tasks_path, expected, destination, start_seed = sys.argv[1:]
expected = int(expected)
start_seed = int(start_seed)
mapping = {}
for task in Path(tasks_path).read_text().splitlines():
    payload = json.loads((Path(progress_root) / f"{task}.json").read_text())
    episodes = payload["episode_results"]
    seeds = [int(item["seed"]) for item in episodes]
    if len(seeds) != expected or any(right <= left for left, right in zip(seeds, seeds[1:])):
        raise RuntimeError(f"{task}: cannot freeze invalid baseline seed list {seeds}")
    mapping[task] = [
        {"seed": int(item["seed"]), "instruction": str(item["instruction"])}
        for item in episodes
    ]
payload = {"schema": "robotwin-expert-valid-seeds-v1", "start_seed": start_seed,
           "episodes_per_task": expected, "tasks": mapping}
temporary = Path(destination + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY
fi

if [[ "$baseline_only_precompute" == true ]]; then
  # This is an audited throughput optimization for a future paired run.  The
  # baseline has already selected and evaluated the exact expert-valid seeds;
  # a later invocation on the same output root re-audits the completed
  # progress, regenerates the identical manifest, and then evaluates ZeVA.
  python3 - "$output/baseline_precompute_complete.json" "$seed_manifest" \
    "$output/baseline/report.json" "$baseline_config_sha256" \
    "$absolute_start_seed" "$expected_total_episodes" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

destination, seed_manifest, report, config_sha256, start_seed, expected = sys.argv[1:]
seed_path = Path(seed_manifest)
report_path = Path(report)
payload = {
    "schema": "zeva-robotwin-baseline-precompute-v1",
    "baseline_config_sha256": config_sha256,
    "absolute_start_seed": int(start_seed),
    "expected_total_episodes": int(expected),
    "seed_manifest": str(seed_path.resolve()),
    "seed_manifest_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
    "baseline_report": str(report_path.resolve()),
    "baseline_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "continuation": "rerun_same_output_root_to_audit_baseline_then_evaluate_zeva",
}
path = Path(destination)
temporary = path.with_name(path.name + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
  printf '{"state":"baseline_precompute_complete","finished":"%s"}\n' \
    "$(date -Iseconds)" > "$output/state.json"
  exit 0
fi

if [[ -n "$anchor_config" ]]; then
  printf '{"state":"running_anchor","started":"%s"}\n' "$(date -Iseconds)" > "$output/state.json"
  run_condition anchor "$anchor_config" "$anchor_label" "$seed_manifest"
fi

printf '{"state":"running_zeva","started":"%s"}\n' "$(date -Iseconds)" > "$output/state.json"
run_condition zeva "$zeva_config" "$zeva_label" "$seed_manifest"

python3 - "$output" "$min_baseline_success_rate" "$baseline_is_untouched_anchor" <<'PY'
import json
import math
import os
import random
import sys
from pathlib import Path

root = Path(sys.argv[1])
minimum_baseline_rate = float(sys.argv[2])
baseline_is_untouched_anchor = sys.argv[3].lower() == "true"
baseline = json.loads((root / "baseline" / "report.json").read_text())
zeva = json.loads((root / "zeva" / "report.json").read_text())
anchor_path = root / "anchor" / "report.json"
anchor = json.loads(anchor_path.read_text()) if anchor_path.is_file() else None
base_rows = {row["task"]: row for row in baseline["tasks"]}
zeva_rows = {row["task"]: row for row in zeva["tasks"]}
paired = []
discordant = {"baseline_only": 0, "zeva_only": 0}
for task in (root / "tasks.txt").read_text().splitlines():
    bp = json.loads((root / "baseline" / "progress" / f"{task}.json").read_text())
    zp = json.loads((root / "zeva" / "progress" / f"{task}.json").read_text())
    bitems, zitems = bp["episode_results"], zp["episode_results"]
    if [x["seed"] for x in bitems] != [x["seed"] for x in zitems]:
        raise RuntimeError(f"{task}: paired seed mismatch")
    if [x["instruction"] for x in bitems] != [x["instruction"] for x in zitems]:
        raise RuntimeError(f"{task}: paired instruction mismatch")
    for b, z in zip(bitems, zitems, strict=True):
        bs, zs = bool(b["success"]), bool(z["success"])
        paired.append(int(zs) - int(bs))
        if bs and not zs:
            discordant["baseline_only"] += 1
        elif zs and not bs:
            discordant["zeva_only"] += 1
n = len(paired)
rng = random.Random(20260903)
bootstrap = []
for _ in range(20000):
    bootstrap.append(sum(paired[rng.randrange(n)] for _ in range(n)) / n)
bootstrap.sort()
d = discordant["baseline_only"] + discordant["zeva_only"]
if d:
    tail = sum(math.comb(d, k) for k in range(min(discordant.values()) + 1)) / (2 ** d)
    mcnemar_p = min(1.0, 2 * tail)
else:
    mcnemar_p = 1.0
task_deltas = []
for task in base_rows:
    task_deltas.append({"task": task,
                        "baseline_success_rate": base_rows[task]["success_rate"],
                        "zeva_success_rate": zeva_rows[task]["success_rate"],
                        "delta": zeva_rows[task]["success_rate"] - base_rows[task]["success_rate"]})
report = {
    "schema": "zeva-robotwin-formal-paired-report-v1",
    "total_paired_episodes": n,
    "baseline_success_rate": baseline["micro_success_rate"],
    "zeva_success_rate": zeva["micro_success_rate"],
    "absolute_delta": zeva["micro_success_rate"] - baseline["micro_success_rate"],
    "anchor_success_rate": None if anchor is None else anchor["micro_success_rate"],
    "baseline_minus_anchor": None if anchor is None else (
        baseline["micro_success_rate"] - anchor["micro_success_rate"]
    ),
    "paired_bootstrap_95ci": [bootstrap[499], bootstrap[19499]],
    "discordant_pairs": discordant,
    "exact_mcnemar_p": mcnemar_p,
    "tasks": task_deltas,
}
temporary = root / "paired_report.json.partial"
temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "paired_report.json")
if anchor is not None or baseline_is_untouched_anchor:
    baseline_rate = float(report["baseline_success_rate"])
    anchor_rate = baseline_rate if baseline_is_untouched_anchor else float(report["anchor_success_rate"])
    zeva_rate = float(report["zeva_success_rate"])
    baseline_floor = max(anchor_rate, minimum_baseline_rate)
    accepted = baseline_rate >= baseline_floor and zeva_rate > baseline_rate
    acceptance = {
        "schema": "zeva-advantage10-acceptance-v1",
        "accepted": accepted,
        "requirements": {
            "baseline_not_below_original_pi_anchor": baseline_rate >= anchor_rate,
            "baseline_not_below_historical_normal_pi": (
                baseline_rate >= minimum_baseline_rate
            ),
            "zeva_strictly_above_trained_baseline": zeva_rate > baseline_rate,
        },
        "success_rates": {
            "anchor": anchor_rate,
            "baseline": baseline_rate,
            "zeva": zeva_rate,
        },
        "minimum_baseline_success_rate": minimum_baseline_rate,
        "effective_baseline_floor": baseline_floor,
        "model_rng_seed": 20260907,
        "model_seed_policy": "continuous",
        "next_action": (
            "deliver" if accepted else "diagnose_and_continue_training_before_delivery"
        ),
    }
    temporary = root / "acceptance.json.partial"
    temporary.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, root / "acceptance.json")
PY

printf '{"state":"complete","finished":"%s"}\n' "$(date -Iseconds)" > "$output/state.json"
