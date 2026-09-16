#!/usr/bin/env bash
# One-shot read-only validation of the published H15-route step500 checkpoint.
# It never waits for files or touches the ongoing four-GPU training process.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_root=${H15_ROUTE_RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/h15-route-full-20260916-a29/h15_route_full}
base=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/fixed-anchor-pair-20260914/baseline/001000
checkpoint=$run_root/000500
manifest=$run_root/manifest.json
context_gate_scale=${H15_CONTEXT_GATE_SCALE:-1}
prior_gate_scale=${H15_PRIOR_GATE_SCALE:-1}
[[ "$context_gate_scale" =~ ^(1|2|4)$ && "$prior_gate_scale" =~ ^(1|10|50)$ ]] || {
  echo "Gate sensitivity supports only fixed context {1,2,4}, prior {1,10,50}" >&2; exit 2;
}
suffix=""
if [[ "$context_gate_scale" != 1 || "$prior_gate_scale" != 1 ]]; then
  suffix="_ctx${context_gate_scale}_prior${prior_gate_scale}"
fi
report=$run_root/validation_diagnostics_000500${suffix}.json
log=$run_root/validation_diagnostics_000500${suffix}.log
expected_uuid=GPU-23cf0ed2-0f7a-f425-7457-781b444b2811
python_bin=${PI05_PYTHON:-/usr/bin/python3}
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
runtime=$handoff/runtime
overlay=${MODEL_DEPENDENCY_OVERLAY:-/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914}
native_transformers=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
shared_deps=/data1/dingxin/zeva-runtime-deps
compiled_deps=/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1

[[ $(hostname -s) == aigc29 ]] || { echo "This diagnostic is pinned to aigc29" >&2; exit 2; }
for required in "$checkpoint/model.safetensors" "$checkpoint/zeva_adapter.pth" \
  "$checkpoint/training_state.pt" "$manifest" "$base/model.safetensors"; do
  [[ -s "$required" ]] || { echo "Checkpoint is not fully published: $required" >&2; exit 2; }
done
[[ ! -e "$report" && ! -e "$log" ]] || { echo "Refusing to overwrite diagnostic output" >&2; exit 2; }
"$python_bin" - "$run_root/latest.json" <<'PY'
import json
from pathlib import Path
import sys
latest = json.loads(Path(sys.argv[1]).read_text())
if int(latest.get("step", -1)) < 500:
    raise SystemExit("Trainer has not published step500 through latest.json")
PY

actual_uuid=$(nvidia-smi -i 4 --query-gpu=uuid --format=csv,noheader)
[[ "$actual_uuid" == "$expected_uuid" ]] || { echo "Diagnostic GPU identity changed" >&2; exit 2; }
gpu_row=$(nvidia-smi -i 4 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
IFS=, read -r used_mib utilization <<< "$gpu_row"
used_mib=${used_mib//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$used_mib" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || exit 2
(( used_mib < 1024 && utilization <= 5 )) || { echo "Diagnostic GPU4 is busy" >&2; exit 2; }
nvidia-smi -i 4 -q -x | "$python_bin" -c '
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.stdin)
others = [(p.findtext("pid"), p.findtext("process_name"))
          for p in root.findall(".//processes/process_info")
          if p.findtext("process_name", "").rsplit("/", 1)[-1] != "Xorg"]
if others:
    raise SystemExit(f"Diagnostic GPU4 has non-Xorg processes: {others}")
'

system_site=$("$python_bin" -c 'import site; print(site.getsitepackages()[0])')
export PYTHONPATH="$overlay:$native_transformers:$shared_deps:$compiled_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export LD_LIBRARY_PATH="$overlay/torch/lib:$overlay/nvidia/cuda_runtime/lib:$overlay/nvidia/cublas/lib:$overlay/nvidia/cudnn/lib:$overlay/nvidia/nccl/lib:$overlay/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$expected_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

"$python_bin" -u "$zeva_root/scripts/eval_robotwin_stage2_diagnostics.py" \
  --checkpoint "$checkpoint" \
  --manifest "$manifest" \
  --output "$report" \
  --fixed-teacher-checkpoint "$base" \
  --batch-size 16 \
  --eval-batches 0 \
  --video-backend torchcodec \
  --decoder-threads 1 \
  --seed 1000 \
  --context-gate-scale "$context_gate_scale" \
  --prior-gate-scale "$prior_gate_scale" \
  > "$log" 2>&1

"$python_bin" - "$report" <<'PY'
import json
from pathlib import Path
import sys
report = json.loads(Path(sys.argv[1]).read_text())
protocol = report["protocol"]
for key, expected in {"complete": True, "validation_decision_samples": 5874,
                      "evaluated_batches": 368, "policy_horizon": 50,
                      "executed_horizon": 15, "checkpoint_written": False,
                      "optimizer_created": False}.items():
    if protocol.get(key) != expected:
        raise SystemExit(f"Incomplete read-only H15 report: {key}={protocol.get(key)!r}")
print("Step500 full read-only H15 diagnostic complete")
PY
