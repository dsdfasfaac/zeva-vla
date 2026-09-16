#!/usr/bin/env bash
# Synthetic H50/H15 serving contract check for the validation-selected model.
# No simulator, test labels, or optimizer are involved.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_root=${H15_ROUTE_RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/h15-route-full-20260916-a29/h15_route_full}
config=$run_root/eval_staging_selected_000500/robotwin_eval_ztev2_zeva.yml
output=$run_root/selected_model_serving_smoke.json
log=$run_root/selected_model_serving_smoke.log
expected_uuid=GPU-23cf0ed2-0f7a-f425-7457-781b444b2811
python_bin=${PI05_PYTHON:-/usr/bin/python3}
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
runtime=$handoff/runtime
overlay=${MODEL_DEPENDENCY_OVERLAY:-/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914}
native_transformers=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
shared_deps=/data1/dingxin/zeva-runtime-deps
compiled_deps=/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1

[[ $(hostname -s) == aigc29 ]] || { echo "Serving smoke is pinned to aigc29" >&2; exit 2; }
[[ -s "$config" && ! -e "$output" && ! -e "$log" ]] || {
  echo "Config absent or smoke output already exists" >&2; exit 2;
}
[[ $(nvidia-smi -i 4 --query-gpu=uuid --format=csv,noheader) == "$expected_uuid" ]] || exit 2
row=$(nvidia-smi -i 4 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
IFS=, read -r used_mib utilization <<< "$row"
used_mib=${used_mib//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$used_mib" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || exit 2
(( used_mib < 1024 && utilization <= 5 )) || { echo "GPU4 is busy" >&2; exit 2; }
nvidia-smi -i 4 -q -x | "$python_bin" -c '
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.stdin)
others = [(p.findtext("pid"), p.findtext("process_name"))
          for p in root.findall(".//processes/process_info")
          if p.findtext("process_name", "").rsplit("/", 1)[-1] != "Xorg"]
if others:
    raise SystemExit(f"GPU4 has non-Xorg processes: {others}")
'

system_site=$($python_bin -c 'import site; print(site.getsitepackages()[0])')
export PYTHONPATH="$overlay:$native_transformers:$shared_deps:$compiled_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export LD_LIBRARY_PATH="$overlay/torch/lib:$overlay/nvidia/cuda_runtime/lib:$overlay/nvidia/cublas/lib:$overlay/nvidia/cudnn/lib:$overlay/nvidia/nccl/lib:$overlay/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$expected_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

"$python_bin" -u "$zeva_root/scripts/robotwin_eval/smoke_selected_model_runtime.py" \
  --config "$config" --output "$output" > "$log" 2>&1
"$python_bin" - "$output" <<'PY'
import json
from pathlib import Path
import sys
d = json.loads(Path(sys.argv[1]).read_text())
if not (d["passed"] and d["first_shape"] == [50, 16]
        and d["second_shape"] == [50, 16]
        and d["committed_horizon"] == 15
        and d["recurrent_transitions"] == 1
        and d["closed_loop_evaluation"] is False):
    raise SystemExit("Selected ZeVA H50/H15 synthetic serving smoke failed")
print("Selected ZeVA H50/H15 synthetic serving smoke passed; no rollout started")
PY
