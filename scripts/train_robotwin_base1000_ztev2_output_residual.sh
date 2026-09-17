#!/usr/bin/env bash
# Frozen Base1000 + ZTE v2 direct H15 residual; cached PI actions, train95 only.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
runtime=$handoff/runtime
overlay=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
native_transformers=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
shared_deps=/data1/dingxin/zeva-runtime-deps
compiled_deps=/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1
run_root=${BASE1000_ZTEV2_RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/base1000-ztev2-output-residual-20260917}
mode=${1:-full}
base=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/fixed-anchor-pair-20260914/baseline/001000
stage1_root=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911
zte=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-phase-vector-mse-4096-20260911h-scheduler-repair-20260911i/zte_v2_step_004096.pth
cache=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/base1000-residual-probe-20260917/base_action_cache.pt
dataset=/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data
expected_uuid=GPU-dc037006-cc5d-131d-9b69-cae335dcf41f

[[ $(hostname -s) == aigc29 ]] || { echo "Pinned to aigc29" >&2; exit 2; }
[[ "$mode" == smoke || "$mode" == full ]] || { echo "Usage: $0 [smoke|full]" >&2; exit 2; }
for required in "$base/model.safetensors" "$zte" "$cache" "$dataset/adapter.json" \
  "$stage1_root/train_causal_bank.pt" "$stage1_root/live_queries_h15.pt" \
  "$stage1_root/task_retrieval.pth" "$zeva_root/configs/robotwin_zeva_advantage10.json"; do
  [[ -s "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
actual_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
[[ "$actual_uuid" == "$expected_uuid" ]] || { echo "GPU0 UUID changed" >&2; exit 2; }
gpu_row=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
IFS=, read -r used_mib utilization <<< "$gpu_row"
used_mib=${used_mib//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
(( used_mib < 1024 && utilization <= 5 )) || { echo "GPU0 is busy" >&2; exit 2; }

output=$run_root/$mode
[[ ! -e "$output/manifest.json" ]] || { echo "Refusing to overwrite $output" >&2; exit 2; }
mkdir -p "$output"
steps=500
save_freq=250
extra_args=()
if [[ "$mode" == smoke ]]; then
  steps=1
  save_freq=1
  extra_args=(--no-save-checkpoints)
fi

python_bin=/usr/bin/python3
system_site=$($python_bin -c 'import site; print(site.getsitepackages()[0])')
export PYTHONPATH="$overlay:$native_transformers:$shared_deps:$compiled_deps:$system_site:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export LD_LIBRARY_PATH="$overlay/torch/lib:$overlay/nvidia/cuda_runtime/lib:$overlay/nvidia/cublas/lib:$overlay/nvidia/cudnn/lib:$overlay/nvidia/nccl/lib:$overlay/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$expected_uuid"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

exec "$python_bin" -u "$zeva_root/scripts/train_robotwin_stage2.py" \
  --handoff-root "$handoff" \
  --foundation-checkpoint "$handoff/checkpoint/pretrained_model-best-v1" \
  --initial-stage2-checkpoint "$base" \
  --goal-embedding-checkpoint "$handoff/checkpoint/pretrained_model" \
  --training-variant output_residual \
  --dataset-root "$dataset" \
  --task-subset "$zeva_root/configs/robotwin_zeva_advantage10.json" \
  --zte-checkpoint "$zte" \
  --causal-bank "$stage1_root/train_causal_bank.pt" \
  --live-queries "$stage1_root/live_queries_h15.pt" \
  --task-retrieval "$stage1_root/task_retrieval.pth" \
  --base-action-cache "$cache" \
  --save-dir "$output" --steps "$steps" --save-freq "$save_freq" \
  --batch-size 256 --gradient-accumulation-steps 1 \
  --num-workers 4 --video-backend torchcodec \
  --prior-injection-horizon 15 \
  --prior-loss-weight 0 --preserve-loss-weight 8 \
  --paired-improvement-margin 0 --gate-regularization-weight 0.0001 \
  --residual-bound 0.2 --residual-regression-weight 0.25 \
  --residual-trust-region-weight 0.1 --residual-trust-region-radius 0.05 \
  --prior-residual-dropout-probability 0 \
  --learning-rate 5e-5 --warmup-steps 50 \
  --eval-batches 1000000 --log-freq 10 --no-compile-model \
  "${extra_args[@]}"
