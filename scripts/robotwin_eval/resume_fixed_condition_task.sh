#!/usr/bin/env bash
set -euo pipefail

# Resume one task against an already-running model server.  Fixed-manifest
# replay is the default; FIXED_SEED_SEQUENCE=false also supports the initial
# Base expert-filtering pass before a manifest exists.  The model server is
# never restarted, so completed progress and the continuous model RNG remain
# aligned.

if (( $# != 8 )); then
  echo "usage: $0 CONDITION_ROOT TASK SLOT PORT MODEL_IP CKPT_LABEL SEED_MANIFEST ZEVA_ROOT" >&2
  exit 64
fi

condition_root=$1
task=$2
slot=$3
port=$4
model_ip=$5
ckpt_label=$6
seed_manifest=$7
zeva_root=$8
render_host=${RENDER_HOST:-aigc24}
render_runtime=${RENDER_RUNTIME:-/data1/dingxin/robotwin-formal-eval/RoboTwin}
render_vulkan_icd=${RENDER_VULKAN_ICD:-/usr/share/vulkan/icd.d/nvidia_icd.json}
render_mps_pipe_directory=${RENDER_MPS_PIPE_DIRECTORY:-}
fixed_seed_sequence=${FIXED_SEED_SEQUENCE:-true}
max_process_attempts=${MAX_PROCESS_ATTEMPTS:-20}
test_num=${TEST_NUM:-20}
absolute_start_seed=${ABSOLUTE_START_SEED:-1000}
log="$condition_root/logs/$task.log"
progress="$condition_root/progress/$task.json"
status="$condition_root/status/$task.json"
rescue_log="$condition_root/logs/$task-rescue.log"

if [[ "$fixed_seed_sequence" != true && "$fixed_seed_sequence" != false ]]; then
  echo "FIXED_SEED_SEQUENCE must be true or false" >&2
  exit 64
fi
if [[ "$fixed_seed_sequence" == true ]]; then
  test -s "$seed_manifest"
  seed_args="--fixed_seed_sequence True --seed_manifest '$seed_manifest'"
else
  seed_args="--fixed_seed_sequence False"
fi
render_mps_env=""
if [[ -n "$render_mps_pipe_directory" ]]; then
  ssh "$render_host" "mkdir -p '$render_mps_pipe_directory'"
  render_mps_env="CUDA_MPS_PIPE_DIRECTORY='$render_mps_pipe_directory'"
fi

mkdir -p "$condition_root"/{logs,progress,status,results}
for attempt in $(seq 1 "$max_process_attempts"); do
  if python3 - "$progress" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raise SystemExit(0 if path.is_file() and json.loads(path.read_text()).get("complete") else 1)
PY
  then
    printf '{"job":"%s","slot":%d,"return_code":0,"recovered":true,"attempts":%d,"finished":"%s"}\n' \
      "$task" "$slot" "$((attempt - 1))" "$(date -Iseconds)" > "$status"
    exit 0
  fi

  printf '%s attempt=%d task=%s\n' "$(date -Iseconds)" "$attempt" "$task" >> "$rescue_log"
  set +e
  ssh "$render_host" "cd '$render_runtime'; env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='$slot' VK_ICD_FILENAMES='$render_vulkan_icd' $render_mps_env PYTHONPATH='$zeva_root/scripts/robotwin_eval:$render_runtime/script:$render_runtime:$render_runtime/policy' \
    .venv_robotwin/bin/python '$zeva_root/scripts/robotwin_eval/eval_policy_client.py' --port '$port' --config '$zeva_root/scripts/robotwin_eval/client_config.yml' \
    --overrides --task_name '$task' --task_config zeva_randomized --test_num '$test_num' \
    --instruction_type seen --seed 0 --absolute_start_seed '$absolute_start_seed' \
    $seed_args \
    --model_seed_policy continuous --policy_name zeva_policy --ckpt_setting '$ckpt_label' \
    --eval_video_log True --result_dir '$condition_root/results/$task' \
    --execute_horizon 15 --chunk_length 50 --action_dim 16 \
    --server_host '$model_ip' --resume_progress_path '$progress'" >> "$log" 2>&1
  rc=$?
  set -e
  printf '%s attempt=%d rc=%d\n' "$(date -Iseconds)" "$attempt" "$rc" >> "$rescue_log"
  if (( rc == 0 )); then
    printf '{"job":"%s","slot":%d,"return_code":0,"recovered":true,"attempts":%d,"finished":"%s"}\n' \
      "$task" "$slot" "$attempt" "$(date -Iseconds)" > "$status"
    exit 0
  fi
  sleep 2
done

printf '{"job":"%s","slot":%d,"return_code":1,"recovered":false,"attempts":%d,"finished":"%s"}\n' \
  "$task" "$slot" "$max_process_attempts" "$(date -Iseconds)" > "$status"
exit 1
