#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for the separately launched scheduler-repair run, select an immutable
# held-out checkpoint, then run the full four-shard export/retrieval workflow.
# This script intentionally never kills or attaches to a training process.

repo_root=${ZEVA_REPO_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA}
run_dir=${ROBOTWIN_ZTE_REPAIR_RUN:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-phase-vector-mse-4096-20260911h-scheduler-repair-20260911i}
training_pid=${ROBOTWIN_ZTE_REPAIR_PID:-672753}
training_log=${ROBOTWIN_ZTE_REPAIR_LOG:-${run_dir}/train.log}
selection_json=${ROBOTWIN_ZTE_SELECTION_OUTPUT:-${run_dir}/checkpoint_selection.json}
output_dir=${ROBOTWIN_ZTE_ARTIFACT_DIR:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911}
waiter_log=${ROBOTWIN_ZTE_WAITER_LOG:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-scheduler-repaired-export-waiter-20260911.log}
selector=${repo_root}/scripts/select_robotwin_zte_v2_scheduler_repaired.py
launcher=${repo_root}/scripts/export_robotwin_zte_v2_full_20260911.sh

if [[ -e "${selection_json}" || -e "${output_dir}" ]]; then
  echo "Refusing to reuse selection/output paths: selection=${selection_json} output=${output_dir}" >&2
  exit 2
fi
if [[ -e "${waiter_log}" ]]; then
  echo "Refusing to append to an existing waiter log: ${waiter_log}" >&2
  exit 2
fi
mkdir -p "$(dirname "${waiter_log}")"
exec > >(tee -a "${waiter_log}") 2>&1

export ZEVA_PROCESSES=1
export ZEVA_RUNTIME_DEPS=${ZEVA_RUNTIME_DEPS:-/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1}
export NATIVE_TRANSFORMERS_RUNTIME=${NATIVE_TRANSFORMERS_RUNTIME:-/data1/dingxin/transformers5-runtime}
export PI05_PYTHON=${PI05_PYTHON:-python3}
export PYTHONUNBUFFERED=1

echo "waiter_start=$(date -Is)"
echo "repair_run=${run_dir} training_pid=${training_pid}"
echo "final_artifacts=${output_dir}"

if [[ ! -f "${selector}" || ! -f "${launcher}" ]]; then
  echo "Selector or launcher is missing." >&2
  exit 2
fi

while :; do
  stat=$(ps -p "${training_pid}" -o stat= 2>/dev/null | tr -d '[:space:]' || true)
  if [[ -z "${stat}" || "${stat}" == *Z* ]]; then
    echo "training_pid=${training_pid} is no longer active (stat=${stat:-gone})"
    break
  fi
  echo "waiting_for_training_pid=${training_pid} stat=${stat} $(date -Is)"
  sleep 30
done

# A launcher can disappear while its accelerate children are still alive.
# Refuse to export in that ambiguous state rather than racing the last worker.
remaining=$(pgrep -af "train_robotwin_zte_v2.py" | grep -F "${run_dir}" || true)
if [[ -n "${remaining}" ]]; then
  echo "Training workers still reference the repair run; refusing export:" >&2
  echo "${remaining}" >&2
  exit 1
fi

final_checkpoint="${run_dir}/zte_v2_step_004096.pth"
if [[ ! -s "${final_checkpoint}" ]]; then
  echo "Required final checkpoint is missing/empty: ${final_checkpoint}" >&2
  exit 1
fi

log_arg=()
if [[ -f "${training_log}" ]]; then
  log_arg=(--training-log "${training_log}")
  echo "training_log=${training_log}"
else
  echo "training_log=${training_log} unavailable; selector will require final checkpoint validation and record this caveat."
fi

echo "selecting_checkpoint=$(date -Is)"
"${PI05_PYTHON}" -m py_compile "${selector}"
"${PI05_PYTHON}" "${selector}" \
  --run-dir "${run_dir}" \
  --output "${selection_json}" \
  --final-step 4096 \
  --training-pid "${training_pid}" \
  "${log_arg[@]}" \
  --allow-missing-log

selected_checkpoint=$("${PI05_PYTHON}" - "${selection_json}" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
print(payload["selected"]["path"])
PY
)
selected_sha=$("${PI05_PYTHON}" - "${selection_json}" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
print(payload["selected"]["sha256"])
PY
)
actual_sha=$(sha256sum "${selected_checkpoint}" | awk '{print $1}')
if [[ "${actual_sha}" != "${selected_sha}" ]]; then
  echo "Selected checkpoint changed after pinning: expected=${selected_sha} actual=${actual_sha}" >&2
  exit 1
fi
echo "selected_checkpoint=${selected_checkpoint}"
echo "selected_checkpoint_sha256=${selected_sha}"

for gpu in 0 1 3 4; do
  compute=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  if [[ -n "${compute}" ]]; then
    echo "GPU ${gpu} is occupied; refusing overlap/termination:" >&2
    echo "${compute}" >&2
    exit 1
  fi
done

if [[ -e "${output_dir}" ]]; then
  echo "Artifact output appeared while waiting; refusing overwrite: ${output_dir}" >&2
  exit 1
fi

echo "starting_final_export=$(date -Is) GPUs=0,1,3,4"
export ROBOTWIN_ZTE_CHECKPOINT="${selected_checkpoint}"
export ROBOTWIN_ZTE_CHECKPOINT_SHA256="${selected_sha}"
export ROBOTWIN_ZTE_ARTIFACT_DIR="${output_dir}"
export ROBOTWIN_ZTE_GPU_LIST="0 1 3 4"
export ROBOTWIN_ZTE_SELECTION_JSON="${selection_json}"
export ROBOTWIN_ZTE_EXPORT_BATCH_SIZE=2
export ROBOTWIN_ZTE_EXPORT_WORKERS=2
bash "${launcher}"
echo "final_export_workflow_complete=$(date -Is)"
