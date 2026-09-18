#!/usr/bin/env bash
# New, isolated BehaviorVLA+effect branch. Never resumes an old adapter/bank.
set -euo pipefail
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-test}
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
runtime=$handoff/runtime
overlay=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
native_transformers=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
compiled_deps=/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1
shared_deps=/data1/dingxin/zeva-runtime-deps
[[ $(hostname -s) == aigc28 ]] || { echo 'This launcher is audited for aigc28 only'; exit 2; }
uuid=GPU-c4d9aceb-c274-fb29-e145-f6c2ef560bdc
read -r actual used utilization < <(nvidia-smi -i 0 --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
[[ "$actual" == "$uuid" && "$used" -lt 1024 && "$utilization" -le 5 ]] || { echo 'GPU is not safely free'; exit 2; }
export CUDA_VISIBLE_DEVICES="$uuid"
export PYTHONPATH="$overlay:$native_transformers:$shared_deps:$compiled_deps:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export LD_LIBRARY_PATH="$overlay/torch/lib:$overlay/nvidia/cuda_runtime/lib:$overlay/nvidia/cublas/lib:$overlay/nvidia/cudnn/lib:$overlay/nvidia/nccl/lib:$overlay/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_HOME=/mnt/100T/users/dingxin/VLA/runtime/zeva-torch-cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
case "$mode" in
  test) exec /usr/bin/python3 -u "$zeva_root/scripts/test_robotwin_behavior_effect.py" ;;
  smoke|stage1)
    run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-behavior-effect-20260918
    extra=()
    if [[ "$mode" == smoke ]]; then
      extra=(--smoke-steps 2)
    else
      extra=(--dataset-identity-report "$run/task-dataset-identity-aigc28.json"
             --source-task-identity-report "$run/task-dataset-identity-aigc29.json")
    fi
    exec /usr/bin/python3 -u "$zeva_root/scripts/train_robotwin_behavior_effect_cte.py" \
      --dataset-root /data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data \
      --save-dir "$run/$mode" --task-subset "$zeva_root/configs/robotwin_zeva_advantage10.json" "${extra[@]}"
    ;;
  *) echo 'Usage: test | smoke | stage1'; exit 2 ;;
esac
