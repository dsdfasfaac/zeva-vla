#!/usr/bin/env bash
set -euo pipefail

# Throughput-only accelerator for an already running anchored-v9 fresh final.
# The primary launcher selects the expert-valid seed manifest with Base, then
# evaluates Anchor.  This helper pauses only the primary coordinator shell
# after all Anchor workers have started; remote Anchor workers keep running.
# It evaluates ZeVA on the exact frozen manifest using disjoint ports/render
# resources, audits the condition, atomically imports it, and resumes the
# primary launcher so the ordinary final report/audit remains authoritative.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-anchored-v9/adapter}
final_root=${FINAL_ROOT:-$eval_root/fresh-final-seed10000}
staging_root=${STAGING_ROOT:-$eval_root/fresh-final-seed10000-zeva-parallel}
primary_pid=${PRIMARY_PAIRED_PID:?PRIMARY_PAIRED_PID must identify the active primary paired launcher}
baseline_config=${BASELINE_CONFIG:-$eval_root/configs/base-step-3000.yml}
zeva_config=${ZEVA_CONFIG:-$eval_root/configs/zeva-step-1250.yml}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
foundation_sha256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
episodes=${EPISODES:-20}
start_seed=${ABSOLUTE_START_SEED:-10000}
poll_seconds=${POLL_SECONDS:-30}
model_host=${MODEL_HOST:-aigc32}
model_ip=${MODEL_IP:-172.16.80.166}
base_port=${BASE_PORT:-19300}
render_host=${RENDER_HOST:-aigc15}
render_runtime=${RENDER_RUNTIME:-/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin}
render_mps_pipe_directory=${RENDER_MPS_PIPE_DIRECTORY:-/tmp/zeva-v9-fresh-final-parallel-mps}
state_file=$final_root/parallel_zeva_acceleration_state.json
paused=false

write_state() {
  local state=$1
  printf '{"state":"%s","updated":"%s","primary_pid":%s}\n' \
    "$state" "$(date -Iseconds)" "$primary_pid" > "$state_file"
}

resume_primary() {
  if [[ "$paused" == true ]] && kill -0 "$primary_pid" 2>/dev/null; then
    kill -CONT "$primary_pid"
    paused=false
  fi
}
trap resume_primary EXIT INT TERM

test -s "$baseline_config"
test -s "$zeva_config"
test -s "$task_manifest"
kill -0 "$primary_pid"
primary_command=$(ps -p "$primary_pid" -o args=)
[[ "$primary_command" == *"launch_paired_formal_eval.sh"* ]] || {
  echo "PRIMARY_PAIRED_PID is not the paired launcher: $primary_command" >&2
  exit 2
}

mkdir -p "$final_root"
write_state waiting_for_frozen_base_and_anchor_workers
while true; do
  if [[ -s "$final_root/baseline/report.json" \
        && -s "$final_root/baseline/state.json" \
        && -s "$final_root/seed_manifest.json" \
        && -s "$final_root/tasks.txt" \
        && -s "$final_root/anchor/state.json" \
        && -s "$final_root/anchor/worker-pids.txt" ]]; then
    read -r base_state < <(python3 - "$final_root/baseline/state.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("state", ""))
PY
)
    read -r anchor_state < <(python3 - "$final_root/anchor/state.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("state", ""))
PY
)
    worker_count=$(wc -l < "$final_root/anchor/worker-pids.txt")
    if [[ "$base_state" == complete && "$anchor_state" == running \
          && "$worker_count" -eq 8 ]]; then
      break
    fi
  fi
  kill -0 "$primary_pid" || {
    write_state primary_launcher_died_before_anchor
    exit 3
  }
  sleep "$poll_seconds"
done

# Stop only the local shell at its wait point.  The eight Anchor worker
# subshells and their remote clients continue independently.
kill -STOP "$primary_pid"
paused=true
write_state primary_paused_anchor_and_parallel_zeva_running

if [[ -e "$staging_root" ]]; then
  echo "refusing to reuse non-empty parallel staging root: $staging_root" >&2
  exit 4
fi
mkdir -p "$staging_root"
cp "$final_root/tasks.txt" "$staging_root/tasks.txt"
cp "$final_root/seed_manifest.json" "$staging_root/seed_manifest.json"
cp -al "$final_root/baseline" "$staging_root/baseline"

python3 - "$staging_root" "$baseline_config" "$start_seed" "$episodes" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root, config, start_seed, episodes = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
seed_path = root / "seed_manifest.json"
report_path = root / "baseline" / "report.json"
report = json.loads(report_path.read_text())
seed_payload = json.loads(seed_path.read_text())
expected = episodes * 10
if report.get("condition") != "baseline" or report.get("total_episodes") != expected:
    raise RuntimeError("primary Base report is incomplete")
if (seed_payload.get("start_seed") != start_seed
        or seed_payload.get("episodes_per_task") != episodes
        or len(seed_payload.get("tasks", {})) != 10):
    raise RuntimeError("primary frozen seed manifest does not match final plan")
payload = {
    "schema": "zeva-robotwin-baseline-precompute-v1",
    "baseline_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "absolute_start_seed": start_seed,
    "expected_total_episodes": expected,
    "seed_manifest": str(seed_path.resolve()),
    "seed_manifest_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
    "baseline_report": str(report_path.resolve()),
    "baseline_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    "continuation": "parallel_fresh_final_zeva_on_exact_primary_manifest",
}
destination = root / "baseline_precompute_complete.json"
temporary = destination.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY

env MODEL_HOST="$model_host" MODEL_IP="$model_ip" BASE_PORT="$base_port" \
  RENDER_HOST="$render_host" RENDER_RUNTIME="$render_runtime" \
  RENDER_MPS_PIPE_DIRECTORY="$render_mps_pipe_directory" \
  OUTPUT_ROOT="$staging_root" BASELINE_CONFIG="$baseline_config" ZEVA_CONFIG="$zeva_config" \
  ANCHOR_CONFIG="" BASELINE_IS_UNTOUCHED_ANCHOR=false \
  REQUIRE_EXPLICIT_FOUNDATION=true FOUNDATION_MODEL_SHA256="$foundation_sha256" \
  TASK_MANIFEST="$task_manifest" EPISODES="$episodes" ABSOLUTE_START_SEED="$start_seed" \
  MODEL_RNG_SEED=20260907 MODEL_SEED_POLICY=continuous MIN_BASELINE_SUCCESS_RATE=0 \
  BASELINE_LABEL=anchored-v9-base-step-3000-fresh-final-parallel \
  ZEVA_LABEL=anchored-v9-zeva-step-1250-fresh-final-parallel \
  REUSE_EXISTING_BASELINE=true \
  bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh" \
  > "$staging_root/launcher.log" 2>&1

python3 - "$final_root" "$staging_root" "$episodes" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

final_root, staging_root, episodes = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
expected = episodes * 10
final_seeds = json.loads((final_root / "seed_manifest.json").read_text())
staging_seeds = json.loads((staging_root / "seed_manifest.json").read_text())
if final_seeds != staging_seeds:
    raise RuntimeError("parallel ZeVA did not use the exact primary seed manifest")
report = json.loads((staging_root / "zeva" / "report.json").read_text())
if report.get("condition") != "zeva" or report.get("total_episodes") != expected:
    raise RuntimeError("parallel ZeVA condition is incomplete")
videos = list((staging_root / "zeva" / "results").glob("**/*.mp4"))
if len(videos) != expected or any(path.stat().st_size == 0 for path in videos):
    raise RuntimeError("parallel ZeVA video audit failed")
for task, frozen in final_seeds["tasks"].items():
    progress = json.loads(
        (staging_root / "zeva" / "progress" / f"{task}.json").read_text()
    )["episode_results"]
    if [row["seed"] for row in progress] != [row["seed"] for row in frozen]:
        raise RuntimeError(f"{task}: parallel ZeVA seed mismatch")
    if [row["instruction"] for row in progress] != [row["instruction"] for row in frozen]:
        raise RuntimeError(f"{task}: parallel ZeVA instruction mismatch")
provenance = {
    "schema": "zeva-robotwin-parallel-final-condition-import-v1",
    "source": str((staging_root / "zeva").resolve()),
    "source_report_sha256": hashlib.sha256(
        (staging_root / "zeva" / "report.json").read_bytes()
    ).hexdigest(),
    "seed_manifest_sha256": hashlib.sha256(
        (final_root / "seed_manifest.json").read_bytes()
    ).hexdigest(),
    "episodes": expected,
    "model_rng_seed": 20260907,
    "model_seed_policy": "continuous",
    "semantic_change": False,
}
destination = final_root / "parallel_zeva_import.json"
temporary = destination.with_suffix(".json.partial")
temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY

test ! -e "$final_root/zeva"
import_tmp="$final_root/zeva.importing.$$"
# Preserve the staging logs as independent inference evidence.  The primary
# launcher will briefly reopen the imported condition to run its normal resume
# audit, which truncates its own worker/server logs.
cp -a "$staging_root/zeva" "$import_tmp"
mv "$import_tmp" "$final_root/zeva"
write_state parallel_zeva_imported_resuming_primary
resume_primary
write_state primary_resumed_for_authoritative_final_audit
