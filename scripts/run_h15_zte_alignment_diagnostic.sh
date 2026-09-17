#!/usr/bin/env bash
# Read-only held-out validation: true ZTE alignment versus within-language-task roll.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_root=${H15_ROUTE_RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/h15-route-full-20260916-a29/h15_route_full}
checkpoint=$run_root/000500
eval_batches=${H15_ZTE_EVAL_BATCHES:-0}
report=${H15_ZTE_ALIGNMENT_REPORT:-$run_root/validation_diagnostics_000500_zte_within_task_roll.json}
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
runtime=$handoff/runtime
overlay=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
native_transformers=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
shared_deps=/data1/dingxin/zeva-runtime-deps
compiled_deps=/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1
expected_uuid=GPU-23cf0ed2-0f7a-f425-7457-781b444b2811

[[ $(hostname -s) == aigc29 ]] || { echo "This diagnostic is pinned to aigc29" >&2; exit 2; }
[[ "$eval_batches" =~ ^[0-9]+$ ]] || exit 2
[[ -s "$checkpoint/model.safetensors" && -s "$checkpoint/zeva_adapter.pth" && -s "$run_root/manifest.json" ]] || exit 2
[[ ! -e "$report" ]] || { echo "Refusing to overwrite $report" >&2; exit 2; }
actual_uuid=$(nvidia-smi -i 4 --query-gpu=uuid --format=csv,noheader)
[[ "$actual_uuid" == "$expected_uuid" ]] || exit 2
gpu_row=$(nvidia-smi -i 4 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
IFS=, read -r used_mib utilization <<< "$gpu_row"
used_mib=${used_mib//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
(( used_mib < 1024 && utilization <= 5 )) || { echo "GPU4 is busy" >&2; exit 2; }

python_bin=/usr/bin/python3
system_site=$($python_bin -c 'import site; print(site.getsitepackages()[0])')
export PYTHONPATH="$overlay:$native_transformers:$shared_deps:$compiled_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export LD_LIBRARY_PATH="$overlay/torch/lib:$overlay/nvidia/cuda_runtime/lib:$overlay/nvidia/cublas/lib:$overlay/nvidia/nccl/lib:$overlay/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$expected_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

exec "$python_bin" -u "$zeva_root/scripts/eval_robotwin_stage2_diagnostics.py" \
  --checkpoint "$checkpoint" --manifest "$run_root/manifest.json" \
  --output "$report" \
  --fixed-teacher-checkpoint /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/fixed-anchor-pair-20260914/baseline/001000 \
  --batch-size 16 --eval-batches "$eval_batches" \
  --video-backend torchcodec --decoder-threads 1 --seed 1000 \
  --zte-ablation within_language_task_roll
