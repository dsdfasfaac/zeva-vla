#!/usr/bin/env bash
# Frozen validation5 matched-noise action diagnostics for the completed epoch40 branch.
set -euo pipefail
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ $(hostname -s) == aigc28 ]] || { echo 'Audited validation launcher requires aigc28' >&2; exit 2; }
run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-behavior-effect-20260918
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
runtime=$handoff/runtime
overlay=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
native_transformers=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
compiled_deps=/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1
shared_deps=/data1/dingxin/zeva-runtime-deps
export PYTHONPATH="$overlay:$native_transformers:$shared_deps:$compiled_deps:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export LD_LIBRARY_PATH="$overlay/torch/lib:$overlay/nvidia/cuda_runtime/lib:$overlay/nvidia/cublas/lib:$overlay/nvidia/cudnn/lib:$overlay/nvidia/nccl/lib:$overlay/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_HOME=/mnt/100T/users/dingxin/VLA/runtime/zeva-torch-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

expected_uuid=GPU-1b146bd7-326f-3563-b97e-24a26e13fa09
read -r actual used util < <(nvidia-smi -i 1 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
[[ "$actual" == "$expected_uuid" && "$used" -lt 1024 && "$util" -le 5 ]] || {
  echo 'GPU1 is not the audited idle H100; leaving other jobs alone' >&2; exit 2;
}
export CUDA_VISIBLE_DEVICES=$expected_uuid
output=$run/validation5-epoch040-exploratory-step5000
[[ ! -e $output ]] || { echo 'Fresh validation output already exists' >&2; exit 2; }
exec /usr/bin/python3 -u "$zeva_root/scripts/report_robotwin_behavior_effect.py" \
  --dataset-root /data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data \
  --cte-checkpoint "$run/stage1/cte_epoch_040.pth" \
  --artifacts "$run/cte-epoch040-exploratory-artifacts.pth" \
  --retrieval-checkpoint /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911/task_retrieval.pth \
  --foundation-checkpoint "$handoff/checkpoint/pretrained_model-best-v1" \
  --checkpoint "$run/stage2-fullpi-pbd-effect-epoch040-exploratory/005000" \
  --output-dir "$output" \
  --expected-decisions "$run/validation5-expected-decisions.json" \
  --plan "$zeva_root/configs/robotwin_behavior_effect_20260918.json" \
  --tasks "$zeva_root/configs/robotwin_zeva_advantage10.json" \
  --handoff "$handoff" --exploratory-epoch40
