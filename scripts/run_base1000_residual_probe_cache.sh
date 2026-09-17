#!/usr/bin/env bash
# Cache Base1000 predictions on train95/validation5 for a ZTE-increment probe.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/base1000-residual-probe-20260917
cache=$output_root/base_action_cache.pt
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
runtime=$handoff/runtime
overlay=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
native_transformers=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
shared_deps=/data1/dingxin/zeva-runtime-deps
compiled_deps=/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1
base=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/fixed-anchor-pair-20260914/baseline/001000
zte=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-phase-vector-mse-4096-20260911h-scheduler-repair-20260911i/zte_v2_step_004096.pth
live=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911/live_queries_h15.pt

[[ $(hostname -s) == aigc29 ]] || { echo "Pinned to aigc29" >&2; exit 2; }
[[ ! -e "$cache" ]] || { echo "Refusing to overwrite $cache" >&2; exit 2; }
for required in "$base/model.safetensors" "$zte" "$live" "$zeva_root/configs/robotwin_zeva_advantage10.json" \
    /data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data/adapter.json; do
  [[ -s "$required" ]] || { echo "Missing required artifact $required" >&2; exit 2; }
done
while IFS=, read -r used utilization; do
  used=${used//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  (( used < 1024 && utilization <= 5 )) || { echo "An aigc29 GPU is busy" >&2; exit 2; }
done < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)

python_bin=/usr/bin/python3
system_site=$($python_bin -c 'import site; print(site.getsitepackages()[0])')
export PYTHONPATH="$overlay:$native_transformers:$shared_deps:$compiled_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export LD_LIBRARY_PATH="$overlay/torch/lib:$overlay/nvidia/cuda_runtime/lib:$overlay/nvidia/cublas/lib:$overlay/nvidia/cudnn/lib:$overlay/nvidia/nccl/lib:$overlay/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=16

exec "$python_bin" -m accelerate.commands.launch --num_machines 1 --num_processes 8 \
  --mixed_precision no "$zeva_root/scripts/cache_robotwin_base_actions_v13.py" \
  --handoff-root "$handoff" \
  --foundation-checkpoint "$handoff/checkpoint/pretrained_model-best-v1" \
  --stage2-checkpoint "$base" \
  --goal-embedding-checkpoint "$handoff/checkpoint/pretrained_model-best-v1" \
  --zte-checkpoint "$zte" --live-queries "$live" \
  --dataset-root /data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data \
  --task-subset "$zeva_root/configs/robotwin_zeva_advantage10.json" \
  --output "$cache" --train-samples 8192 --validation-samples 4096 \
  --batch-size 16 --num-workers 2 --seed 1000
