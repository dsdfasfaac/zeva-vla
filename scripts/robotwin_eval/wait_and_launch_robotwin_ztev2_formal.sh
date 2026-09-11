#!/usr/bin/env bash
set -euo pipefail

# Durable post-training handoff for the matched RoboTwin v2 pair. This runs on
# the model host under nohup/supervision: it waits for both explicit launchers,
# invokes the immutable held-out selector, stages configs from selected paths,
# and only then enters the existing paired formal evaluator. It never guesses
# checkpoints and never kills a process.

usage() {
  cat >&2 <<'EOF'
usage: wait_and_launch_robotwin_ztev2_formal.sh \
  --zeva-root PATH --train-root PATH --output-root PATH \
  --handoff-root PATH --foundation-checkpoint PATH \
  --goal-embedding-checkpoint PATH --zte-checkpoint PATH \
  --causal-bank PATH --retrieval-checkpoint PATH \
  --task-manifest PATH --base-pid PID --zeva-pid PID \
  [--model-host HOST] [--render-host HOST] [--shared-runtime PATH] \
  [--render-runtime PATH] [--render-vulkan-icd PATH] [--base-port PORT] \
  [--poll-seconds N] [--foundation-model-sha256 SHA256] [--dry-run]

All checkpoint/artifact paths and both launcher PIDs are required. --dry-run
only prints the planned handoff and performs no wait, write, SSH, selector, or
evaluation action.
EOF
  exit 2
}

zeva_root=""
train_root=""
output_root=""
handoff_root=""
foundation_checkpoint=""
goal_embedding_checkpoint=""
zte_checkpoint=""
causal_bank=""
retrieval_checkpoint=""
task_manifest=""
base_pid=""
zeva_pid=""
model_host="aigc29"
render_host="aigc24"
shared_runtime="/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin"
render_runtime="/data1/dingxin/robotwin-formal-eval/RoboTwin"
render_vulkan_icd="/usr/share/vulkan/icd.d/nvidia_icd.json"
base_port=19200
poll_seconds=30
foundation_model_sha256="7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe"
dry_run=false

while (($#)); do
  case "$1" in
    --zeva-root) zeva_root=${2:?missing value for $1}; shift 2 ;;
    --train-root) train_root=${2:?missing value for $1}; shift 2 ;;
    --output-root) output_root=${2:?missing value for $1}; shift 2 ;;
    --handoff-root) handoff_root=${2:?missing value for $1}; shift 2 ;;
    --foundation-checkpoint) foundation_checkpoint=${2:?missing value for $1}; shift 2 ;;
    --goal-embedding-checkpoint) goal_embedding_checkpoint=${2:?missing value for $1}; shift 2 ;;
    --zte-checkpoint) zte_checkpoint=${2:?missing value for $1}; shift 2 ;;
    --causal-bank) causal_bank=${2:?missing value for $1}; shift 2 ;;
    --retrieval-checkpoint) retrieval_checkpoint=${2:?missing value for $1}; shift 2 ;;
    --task-manifest) task_manifest=${2:?missing value for $1}; shift 2 ;;
    --base-pid) base_pid=${2:?missing value for $1}; shift 2 ;;
    --zeva-pid) zeva_pid=${2:?missing value for $1}; shift 2 ;;
    --model-host) model_host=${2:?missing value for $1}; shift 2 ;;
    --render-host) render_host=${2:?missing value for $1}; shift 2 ;;
    --shared-runtime) shared_runtime=${2:?missing value for $1}; shift 2 ;;
    --render-runtime) render_runtime=${2:?missing value for $1}; shift 2 ;;
    --render-vulkan-icd) render_vulkan_icd=${2:?missing value for $1}; shift 2 ;;
    --base-port) base_port=${2:?missing value for $1}; shift 2 ;;
    --poll-seconds) poll_seconds=${2:?missing value for $1}; shift 2 ;;
    --foundation-model-sha256) foundation_model_sha256=${2:?missing value for $1}; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

required_args=(
  zeva_root train_root output_root handoff_root foundation_checkpoint
  goal_embedding_checkpoint zte_checkpoint causal_bank retrieval_checkpoint
  task_manifest base_pid zeva_pid
)
for name in "${required_args[@]}"; do
  [[ -n "${!name}" ]] || { echo "missing required --${name//_/-}" >&2; usage; }
done
[[ "$base_pid" =~ ^[1-9][0-9]*$ ]] || { echo "--base-pid must be a positive PID" >&2; exit 2; }
[[ "$zeva_pid" =~ ^[1-9][0-9]*$ ]] || { echo "--zeva-pid must be a positive PID" >&2; exit 2; }
[[ "$base_port" =~ ^[0-9]+$ ]] || { echo "--base-port must be an integer" >&2; exit 2; }
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || { echo "--poll-seconds must be positive" >&2; exit 2; }
[[ "$foundation_model_sha256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "--foundation-model-sha256 must be a 64-character lowercase SHA256" >&2
  exit 2
}

if [[ "$dry_run" == true ]]; then
  printf '%s\n' "dry-run: no wait, write, SSH, selector, or formal evaluation will run"
  printf 'zeva_root=%s\ntrain_root=%s\noutput_root=%s\nbase_pid=%s\nzeva_pid=%s\n' \
    "$zeva_root" "$train_root" "$output_root" "$base_pid" "$zeva_pid"
  printf 'formal_contract=episodes20,start_seed1000,task_manifest=%s,slots8,base_port=%s\n' \
    "$task_manifest" "$base_port"
  exit 0
fi

base_run="$train_root/baseline"
zeva_run="$train_root/zeva"
selection_json="$output_root/checkpoint_selection.json"
staging_root="$output_root/staging"
pipeline_log="$output_root/pipeline.log"
pipeline_state="$output_root/pipeline_state.json"

if [[ -e "$output_root" ]]; then
  if [[ ! -d "$output_root" ]] || [[ -n "$(find "$output_root" -mindepth 1 -print -quit)" ]]; then
    echo "refusing non-empty existing output root: $output_root" >&2
    exit 2
  fi
fi
mkdir -p "$output_root"
exec > >(tee -a "$pipeline_log") 2>&1

write_state() {
  local state=$1
  local detail=${2:-}
  python3 - "$pipeline_state" "$state" "$detail" <<'PY'
import json
import os
import sys
from pathlib import Path

destination, state, detail = sys.argv[1:]
payload = {
    "schema": "zeva-robotwin-ztev2-posttraining-pipeline-v1",
    "state": state,
    "detail": detail,
}
path = Path(destination)
temporary = path.with_name(path.name + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

fail() {
  local message=$1
  write_state failed "$message" || true
  echo "pipeline failed: $message" >&2
  exit 1
}

trap 'rc=$?; if (( rc != 0 )); then write_state failed "exit_code_${rc}" || true; fi' EXIT

pid_running() {
  local pid=$1
  local stat_path="/proc/$pid/stat"
  [[ -r "$stat_path" ]] || return 1
  local process_state
  process_state=$(awk '{print $3}' "$stat_path" 2>/dev/null || true)
  [[ -n "$process_state" && "$process_state" != Z ]]
}

check_pid_identity() {
  local pid=$1
  local label=$2
  if ! pid_running "$pid"; then
    return 0
  fi
  local command_line
  command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
  [[ "$command_line" == *train_robotwin_stage2.py* ]] || {
    fail "$label PID $pid is a different live process: $command_line"
  }
  [[ "$command_line" == *"$train_root"* ]] || {
    fail "$label PID $pid does not belong to requested train root: $command_line"
  }
}

training_children_live() {
  # Match Python executables, not this awk's own command line (which contains
  # both the root and the training filename and otherwise matches itself).
  ps -eo pid=,comm=,args= 2>/dev/null | awk -v root="$train_root" \
    '$2 ~ /^python([0-9.]+)?$/ && index($0, root) && index($0, "train_robotwin_stage2.py") {print $1}'
}

checkpoint_complete() {
  local run_root=$1
  local role=$2
  [[ -s "$run_root/manifest.json" && -s "$run_root/latest.json" ]] || return 1
  local latest_step
  latest_step=$(python3 - "$run_root/latest.json" <<'PY'
import json
import sys
try:
    print(int(json.load(open(sys.argv[1], encoding="utf-8"))["step"]))
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
    print(-1)
PY
)
  (( latest_step >= 5000 )) || return 1
  for step in $(seq 500 500 5000); do
    local checkpoint="$run_root/$(printf '%06d' "$step")"
    [[ -s "$checkpoint/model.safetensors" && -s "$checkpoint/training_state.pt" ]] || return 1
    if [[ "$role" == zeva ]]; then
      [[ -s "$checkpoint/zeva_adapter.pth" ]] || return 1
    fi
  done
}

check_idle_gpus() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '
    {gsub(/[[:space:]]/, "", $2); if (($2 + 0) > 1024) {print "GPU " $1 " has " $2 " MiB in use" > "/dev/stderr"; bad=1}}
    END {exit bad+0}
  '
}

wait_for_idle_gpus() {
  for attempt in $(seq 1 4); do
    if check_idle_gpus; then
      return 0
    fi
    write_state waiting_for_idle_gpus "attempt=${attempt}/4"
    sleep "$poll_seconds"
  done
  fail "one or more model GPUs are still occupied after launcher shutdown"
}

check_model_ports_free() {
  command -v ss >/dev/null 2>&1 || fail "ss is unavailable"
  for offset in $(seq 0 7); do
    local port=$((base_port + offset))
    if ss -ltn 2>/dev/null | awk -v port=":$port" '$4 ~ port"$" {found=1} END {exit found ? 0 : 1}'; then
      fail "model port $port is already listening"
    fi
  done
}

check_render_runtime() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$render_host" \
    "test -x '$render_runtime/.venv_robotwin/bin/python' && test -f '$render_runtime/task_config/_eval_step_limit.yml' && test -f '$render_vulkan_icd'" \
    || fail "renderer host/runtime preflight failed on $render_host"
}

write_state waiting_for_training "waiting_for_base_and_zeva_pids_and_complete_5000_checkpoints"
echo "waiting for Base PID $base_pid and ZeVA PID $zeva_pid; poll=${poll_seconds}s"
post_exit_polls=0
while true; do
  check_pid_identity "$base_pid" Base
  check_pid_identity "$zeva_pid" ZeVA
  base_live=false
  zeva_live=false
  pid_running "$base_pid" && base_live=true
  pid_running "$zeva_pid" && zeva_live=true
  child_pids=$(training_children_live || true)
  if [[ "$base_live" == false && "$zeva_live" == false && -z "$child_pids" ]]; then
    if checkpoint_complete "$base_run" baseline && checkpoint_complete "$zeva_run" zeva; then
      break
    fi
    post_exit_polls=$((post_exit_polls + 1))
    if (( post_exit_polls >= 4 )); then
      fail "both launchers exited but complete 5000-step checkpoint sets were not found"
    fi
  else
    post_exit_polls=0
  fi
  write_state waiting_for_training "base_live=${base_live};zeva_live=${zeva_live};training_children=$(tr '\n' ',' <<<"$child_pids");post_exit_polls=${post_exit_polls}"
  sleep "$poll_seconds"
done

write_state training_complete "both_launchers_stopped_and_all_checkpoints_verified"
wait_for_idle_gpus
check_model_ports_free
check_render_runtime

write_state selecting_checkpoints "held_out_validation_flow_only"
python3 "$zeva_root/scripts/robotwin_eval/select_robotwin_ztev2_pair.py" \
  --base-dir "$base_run" --zeva-dir "$zeva_run" \
  --base-pid "$base_pid" --zeva-pid "$zeva_pid" \
  --output "$selection_json"

readarray -t selected_paths < <(python3 - "$selection_json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["runs"]["baseline"]["selected"]["checkpoint"])
print(payload["runs"]["zeva"]["selected"]["checkpoint"])
PY
)
[[ ${#selected_paths[@]} -eq 2 ]] || fail "selector did not return two selected checkpoint paths"
selected_base=${selected_paths[0]}
selected_zeva=${selected_paths[1]}
[[ -s "$selected_base/model.safetensors" ]] || fail "selected Base model is missing"
[[ -s "$selected_zeva/model.safetensors" && -s "$selected_zeva/zeva_adapter.pth" ]] || fail "selected ZeVA model or adapter is missing"

write_state staging_configs "selected_paths_explicit_and_preflight_started"
[[ ! -e "$staging_root" ]] || fail "refusing existing staging directory: $staging_root"
python3 "$zeva_root/scripts/robotwin_eval/prepare_robotwin_eval_ztev2.py" \
  --handoff-root "$handoff_root" \
  --foundation-checkpoint "$foundation_checkpoint" \
  --anchor-foundation-checkpoint "$foundation_checkpoint" \
  --goal-embedding-checkpoint "$goal_embedding_checkpoint" \
  --base-stage2-checkpoint "$selected_base" \
  --zeva-stage2-checkpoint "$selected_zeva" \
  --zte-checkpoint "$zte_checkpoint" \
  --causal-bank "$causal_bank" \
  --retrieval-checkpoint "$retrieval_checkpoint" \
  --task-manifest "$task_manifest" \
  --output-dir "$staging_root"

python3 - "$staging_root/robotwin_eval_ztev2_staging_manifest.json" "$selected_base" "$selected_zeva" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
expected_base, expected_zeva = sys.argv[2:]
if manifest.get("status") != "staged_not_evaluated" or manifest.get("formal_eval_started") is not False:
    raise SystemExit("staging manifest is not preparation-only")
inputs = manifest.get("inputs", {})
if inputs.get("base_stage2", {}).get("path") != str(Path(expected_base).resolve()):
    raise SystemExit("staging selected Base path mismatch")
if inputs.get("zeva_stage2", {}).get("path") != str(Path(expected_zeva).resolve()):
    raise SystemExit("staging selected ZeVA path mismatch")
PY

write_state formal_evaluation_started "launching_existing_paired_evaluator_after_all_guards"
OUTPUT_ROOT="$output_root" MODEL_HOST="$model_host" RENDER_HOST="$render_host" \
SHARED_RUNTIME="$shared_runtime" RENDER_RUNTIME="$render_runtime" \
RENDER_VULKAN_ICD="$render_vulkan_icd" BASE_PORT="$base_port" \
SLOTS=8 EPISODES=20 ABSOLUTE_START_SEED=1000 \
MODEL_SEED_POLICY=continuous MODEL_RNG_SEED=20260907 \
BASELINE_IS_UNTOUCHED_ANCHOR=false REQUIRE_EXPLICIT_FOUNDATION=true \
FOUNDATION_MODEL_SHA256="$foundation_model_sha256" TASK_MANIFEST="$task_manifest" \
BASELINE_CONFIG="$staging_root/robotwin_eval_ztev2_baseline.yml" \
ZEVA_CONFIG="$staging_root/robotwin_eval_ztev2_zeva.yml" \
ANCHOR_CONFIG="$staging_root/robotwin_eval_ztev2_anchor.yml" \
BASELINE_LABEL="normal-trained-base-ztev2" ZEVA_LABEL="corrected-ztev2" \
ANCHOR_LABEL="untouched-best-v1-anchor" \
bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

write_state complete "paired_formal_evaluation_completed"
