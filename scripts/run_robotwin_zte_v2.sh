#!/usr/bin/env bash
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
runtime=$handoff/runtime
python_bin=${PI05_PYTHON:-python3}
zeva_deps=${ZEVA_RUNTIME_DEPS:-/data1/dingxin/zeva-runtime-deps}
native_transformers=${NATIVE_TRANSFORMERS_RUNTIME:-/data1/dingxin/transformers5-runtime}
system_site=$($python_bin -c 'import site; print(site.getsitepackages()[0])')
processes=${ZEVA_PROCESSES:-1}

test -d "$native_transformers/transformers"
export EGOSCALE_LEROBOT_SOURCE="$runtime/lerobot-main-py311-v1/src"
export PYTHONPATH="$native_transformers:$zeva_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$EGOSCALE_LEROBOT_SOURCE:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-32}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

"$python_bin" -c 'import torch, mamba_ssm, torchcodec; print("ZTE runtime:", torch.__version__, mamba_ssm.__version__, torchcodec.__version__, flush=True)'
if [[ "${1:-}" == "--test" ]]; then
  shift
  exec "$python_bin" -m unittest "$@"
fi
if [[ "$processes" -eq 1 ]]; then
  exec "$python_bin" "$zeva_root/scripts/train_robotwin_zte_v2.py" "$@"
fi
exec "$python_bin" -m accelerate.commands.launch --multi_gpu \
  --num_machines 1 --num_processes "$processes" --mixed_precision no \
  "$zeva_root/scripts/train_robotwin_zte_v2.py" "$@"
