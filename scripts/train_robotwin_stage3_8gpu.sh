#!/usr/bin/env bash
set -euo pipefail

handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
runtime=${ROBOTWIN_RUNTIME:-$handoff/runtime}
python_bin=${PI05_PYTHON:-python3}
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
zeva_deps=${ZEVA_RUNTIME_DEPS:-/data1/dingxin/zeva-runtime-deps}
system_site=$($python_bin -c 'import site; print(site.getsitepackages()[0])')

export EGOSCALE_LEROBOT_SOURCE=${EGOSCALE_LEROBOT_SOURCE:-$runtime/lerobot-main-py311-v1/src}
export PYTHONPATH="$zeva_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-main-deps-py311-v1:$runtime/lerobot-overlay-v2:$EGOSCALE_LEROBOT_SOURCE:$runtime/src:$zeva_root/src:$zeva_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
num_processes=${NUM_PROCESSES:-8}

ffmpeg -hide_banner -decoders 2>/dev/null | grep libdav1d >/dev/null

exec "$python_bin" -m accelerate.commands.launch \
  --num_machines 1 \
  --num_processes "$num_processes" \
  --mixed_precision no \
  "$zeva_root/scripts/train_robotwin_stage3.py" \
  "$@"
