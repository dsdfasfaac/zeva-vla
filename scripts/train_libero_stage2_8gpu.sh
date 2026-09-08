#!/usr/bin/env bash
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ZEVA_PYTHON:-python3}
mamba_deps=${ZEVA_RUNTIME_DEPS:-/data1/dingxin/zeva-runtime-deps}
pi_deps=${LIBERO_RUNTIME_DEPS:-/data1/dingxin/libero-runtime-deps}
export PYTHONPATH="$pi_deps:$mamba_deps:$zeva_root/src:$zeva_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

cd "$zeva_root"
exec "$python_bin" -m accelerate.commands.launch \
  --num_machines 1 --num_processes 8 --mixed_precision no \
  "$zeva_root/scripts/train_libero_stage2.py" "$@"
