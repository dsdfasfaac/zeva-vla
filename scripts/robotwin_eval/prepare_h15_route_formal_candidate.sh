#!/usr/bin/env bash
# Stage immutable formal-evaluation configs only after the validation selector.
# This script does not start rollout workers or consume test success labels.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_root=${H15_ROUTE_RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/h15-route-full-20260916-a29/h15_route_full}
selection=$run_root/h15_candidate_selection.json
output=$run_root/eval_staging_selected_000500
base=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/fixed-anchor-pair-20260914/baseline/001000
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
foundation=$handoff/checkpoint/pretrained_model-best-v1
goal=$handoff/checkpoint/pretrained_model
stage1=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-phase-vector-mse-4096-20260911h-scheduler-repair-20260911i/zte_v2_step_004096.pth
artifacts=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911
bank=$artifacts/train_causal_bank.pt
retrieval=$artifacts/task_retrieval.pth
tasks=$zeva_root/configs/robotwin_zeva_advantage10.json
python_bin=${PI05_PYTHON:-/usr/bin/python3}

[[ $(hostname -s) == aigc29 ]] || { echo "Staging is pinned to aigc29" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Refusing existing staging output: $output" >&2; exit 2; }
selected=$($python_bin - "$selection" "$run_root/000500" <<'PY'
import json
from pathlib import Path
import sys
selection = json.loads(Path(sys.argv[1]).read_text())
expected = str(Path(sys.argv[2]).resolve())
if (selection.get("decision") != "selected"
        or selection.get("selected_step") != 500
        or selection.get("selected_checkpoint") != expected
        or selection.get("test_success_labels_used") is not False):
    raise SystemExit("No validated step500 H15 candidate")
print(expected)
PY
)
for required in "$selected/model.safetensors" "$selected/zeva_adapter.pth" \
  "$base/model.safetensors" "$stage1" "$bank" "$retrieval" "$tasks"; do
  [[ -s "$required" ]] || { echo "Missing selected formal input: $required" >&2; exit 2; }
done

runtime=$handoff/runtime
overlay=${MODEL_DEPENDENCY_OVERLAY:-/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914}
native_transformers=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
shared_deps=/data1/dingxin/zeva-runtime-deps
compiled_deps=/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1
system_site=$($python_bin -c 'import site; print(site.getsitepackages()[0])')
export PYTHONPATH="$overlay:$native_transformers:$shared_deps:$compiled_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export LD_LIBRARY_PATH="$overlay/torch/lib:$overlay/nvidia/cuda_runtime/lib:$overlay/nvidia/cublas/lib:$overlay/nvidia/cudnn/lib:$overlay/nvidia/nccl/lib:$overlay/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

"$python_bin" "$zeva_root/scripts/robotwin_eval/prepare_robotwin_eval_ztev2.py" \
  --handoff-root "$handoff" \
  --foundation-checkpoint "$foundation" \
  --anchor-foundation-checkpoint "$foundation" \
  --goal-embedding-checkpoint "$goal" \
  --base-stage2-checkpoint "$base" \
  --zeva-stage2-checkpoint "$selected" \
  --zte-checkpoint "$stage1" \
  --causal-bank "$bank" \
  --retrieval-checkpoint "$retrieval" \
  --task-manifest "$tasks" \
  --output-dir "$output"

"$python_bin" - "$output/robotwin_eval_ztev2_staging_manifest.json" "$selected" <<'PY'
import json
from pathlib import Path
import sys
manifest = json.loads(Path(sys.argv[1]).read_text())
if manifest["inputs"]["zeva_stage2"]["path"] != sys.argv[2]:
    raise SystemExit("Staging selected a different ZeVA checkpoint")
print("H15 selected candidate formal configs staged; no rollout started")
PY
