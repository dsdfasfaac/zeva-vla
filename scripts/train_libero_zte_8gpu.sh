#!/usr/bin/env bash
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ZEVA_PYTHON:-python3}
runtime_deps=${ZEVA_RUNTIME_DEPS:-/data1/dingxin/zeva-runtime-deps}
export PYTHONPATH="$runtime_deps:$zeva_root/src:$zeva_root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

cd "$zeva_root"
exec "$python_bin" -m accelerate.commands.launch \
  --num_machines 1 \
  --num_processes 8 \
  --mixed_precision no \
  "$zeva_root/scripts/train_libero_zte.py" "$@"
