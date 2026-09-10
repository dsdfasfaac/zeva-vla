#!/usr/bin/env bash
set -euo pipefail

# v15 is an offline upper-bound probe: cache K=4 consecutive frozen PI0.5
# H50 draws and normalized expert targets.  It does not train a ranker or
# change deployment behavior.

handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
runtime=${ROBOTWIN_RUNTIME:-$handoff/runtime}
python_bin=${PI05_PYTHON:-python3}
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
zeva_deps=${ZEVA_RUNTIME_DEPS:-/data1/dingxin/zeva-runtime-deps}
system_site=$($python_bin -c 'import site; print(site.getsitepackages()[0])')
native_transformers=${NATIVE_TRANSFORMERS_RUNTIME:-/data1/dingxin/transformers5-runtime}

mkdir -p "$native_transformers"
ln -sfn "$runtime/lerobot-main-deps-py311-v1/transformers" "$native_transformers/transformers"
ln -sfn "$runtime/lerobot-main-deps-py311-v1/transformers-5.5.4.dist-info" "$native_transformers/transformers-5.5.4.dist-info"
ln -sfn "$runtime/lerobot-main-deps-py311-v1/huggingface_hub" "$native_transformers/huggingface_hub"
ln -sfn "$runtime/lerobot-main-deps-py311-v1/huggingface_hub-1.27.0.dist-info" "$native_transformers/huggingface_hub-1.27.0.dist-info"
ln -sfn "$runtime/lerobot-main-deps-py311-v1/tokenizers-0.22.2.dist-info" "$native_transformers/tokenizers-0.22.2.dist-info"

export PYTHONPATH="$native_transformers:$zeva_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-32}
processes=${NUM_PROCESSES:-8}

exec "$python_bin" -m accelerate.commands.launch \
  --num_machines 1 \
  --num_processes "$processes" \
  --mixed_precision no \
  "$zeva_root/scripts/cache_robotwin_pi_candidates_v15.py" \
  "$@"
