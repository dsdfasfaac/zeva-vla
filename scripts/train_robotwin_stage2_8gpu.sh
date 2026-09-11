#!/usr/bin/env bash
set -euo pipefail

handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
runtime=${ROBOTWIN_RUNTIME:-$handoff/runtime}
python_bin=${PI05_PYTHON:-python3}
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
zeva_deps=${ZEVA_RUNTIME_DEPS:-/data1/dingxin/zeva-runtime-deps}
system_site=$($python_bin -c 'import site; print(site.getsitepackages()[0])')
native_transformers=${NATIVE_TRANSFORMERS_RUNTIME:-/data1/dingxin/transformers5-runtime}

# PI0.5 was released and validated with Transformers 5.5.4.  Keep its
# pure-Python implementation ahead of the Python-3.10 dependency overlay while
# retaining the host-compatible compiled wheels (Torch, tokenizers, regex).
mkdir -p "$native_transformers"
ln -sfn "$runtime/lerobot-main-deps-py311-v1/transformers" "$native_transformers/transformers"
ln -sfn "$runtime/lerobot-main-deps-py311-v1/transformers-5.5.4.dist-info" "$native_transformers/transformers-5.5.4.dist-info"
ln -sfn "$runtime/lerobot-main-deps-py311-v1/huggingface_hub" "$native_transformers/huggingface_hub"
ln -sfn "$runtime/lerobot-main-deps-py311-v1/huggingface_hub-1.27.0.dist-info" "$native_transformers/huggingface_hub-1.27.0.dist-info"
ln -sfn "$runtime/lerobot-main-deps-py311-v1/tokenizers-0.22.2.dist-info" "$native_transformers/tokenizers-0.22.2.dist-info"

export EGOSCALE_LEROBOT_SOURCE=${EGOSCALE_LEROBOT_SOURCE:-$runtime/lerobot-main-py311-v1/src}
export PYTHONPATH="$native_transformers:$zeva_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$EGOSCALE_LEROBOT_SOURCE:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-32}
# Triton's tiny launcher extension links against the CUDA development name
# ``libcuda.so``. The actual kernels still load the host's real driver library.
cuda_stub=${CUDA_STUB_LIBRARY:-/usr/local/cuda/lib64/stubs}
export LIBRARY_PATH="$cuda_stub${LIBRARY_PATH:+:$LIBRARY_PATH}"

if [[ " $* " == *" --video-backend ffmpeg "* ]]; then
  ffmpeg -hide_banner -decoders 2>/dev/null | grep libdav1d >/dev/null
else
  "$python_bin" -c 'import torchcodec'
fi

exec "$python_bin" -m accelerate.commands.launch \
  --multi_gpu \
  --main_process_port "${ZEVA_MAIN_PROCESS_PORT:-29500}" \
  --num_machines 1 \
  --num_processes "${ZEVA_PROCESSES:-8}" \
  --mixed_precision no \
  "$zeva_root/scripts/train_robotwin_stage2.py" \
  "$@"
