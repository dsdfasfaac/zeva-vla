#!/usr/bin/env bash
set -euo pipefail

handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
runtime=${ROBOTWIN_RUNTIME:-$handoff/runtime}
python_bin=${ZTE_PYTHON:-python3}
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
zeva_deps=${ZEVA_RUNTIME_DEPS:-/data1/dingxin/zeva-runtime-deps}
system_site=$($python_bin -c 'import site; print(site.getsitepackages()[0])')

export PYTHONPATH="$zeva_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-main-deps-py311-v1:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

"$python_bin" -c 'import mamba_ssm, torch; print(torch.__version__, mamba_ssm.__version__)'
ffmpeg -hide_banner -decoders 2>/dev/null | grep libdav1d >/dev/null

exec "$python_bin" -m accelerate.commands.launch \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision no \
  "$zeva_root/scripts/train_robotwin_zte.py" \
  "$@"
