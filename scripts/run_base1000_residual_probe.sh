#!/usr/bin/env bash
# Compare held-out residual predictability with/without frozen ZTE features.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/base1000-residual-probe-20260917
cache=$output_root/base_action_cache.pt
report=${BASE1000_PROBE_REPORT:-$output_root/base1000_zte_residual_probe.json}
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
runtime=$handoff/runtime
overlay=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
native_transformers=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
shared_deps=/data1/dingxin/zeva-runtime-deps
compiled_deps=/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1
live=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911/live_queries_h15.pt
expected_uuid=GPU-dc037006-cc5d-131d-9b69-cae335dcf41f

[[ $(hostname -s) == aigc29 ]] || { echo "Pinned to aigc29" >&2; exit 2; }
[[ -s "$cache" && -s "$live" ]] || { echo "Cache/live queries are not ready" >&2; exit 2; }
[[ ! -e "$report" ]] || { echo "Refusing to overwrite $report" >&2; exit 2; }
actual_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
[[ "$actual_uuid" == "$expected_uuid" ]] || exit 2
gpu_row=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
IFS=, read -r used_mib utilization <<< "$gpu_row"
used_mib=${used_mib//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
(( used_mib < 1024 && utilization <= 5 )) || { echo "GPU0 is busy" >&2; exit 2; }

python_bin=/usr/bin/python3
system_site=$($python_bin -c 'import site; print(site.getsitepackages()[0])')
export PYTHONPATH="$overlay:$native_transformers:$shared_deps:$compiled_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export LD_LIBRARY_PATH="$overlay/torch/lib:$overlay/nvidia/cuda_runtime/lib:$overlay/nvidia/cublas/lib:$overlay/nvidia/cudnn/lib:$overlay/nvidia/nccl/lib:$overlay/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$expected_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

exec "$python_bin" -u "$zeva_root/scripts/probe_robotwin_base_residual_value.py" \
  --handoff-root "$handoff" \
  --dataset-root /data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data \
  --live-queries "$live" \
  --task-subset "$zeva_root/configs/robotwin_zeva_advantage10.json" \
  --base-action-cache "$cache" --output "$report" \
  --steps 1000 --batch-size 512 --learning-rate 3e-4 --seed 1000 --device cuda
