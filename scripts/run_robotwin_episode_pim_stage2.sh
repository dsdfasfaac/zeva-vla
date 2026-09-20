#!/usr/bin/env bash
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ $(hostname -s) == aigc28 ]] || { echo 'Audited Episode-PIM run is restricted to aigc28' >&2; exit 2; }

run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-episode-pim-20260920
parent_run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-behavior-effect-20260918
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
dataset=/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data
cte=$parent_run/stage1/cte_epoch_040.pth
cte_artifacts=$parent_run/cte-epoch040-exploratory-artifacts.pth
retrieval=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911/task_retrieval.pth
foundation=$handoff/checkpoint/pretrained_model-best-v1
parent=$parent_run/stage2-fullpi-pbd-effect-epoch040-exploratory/005000
output=$run/stage2-cte-bit-episode-pim-eap-fixed2000-8gpu

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

for path in "$dataset/adapter.json" "$cte" "$cte_artifacts" "$retrieval" \
            "$foundation/model.safetensors" "$parent/model.safetensors" "$parent/zeva_adapter.pth" "$parent/COMPLETE"; do
  [[ -f $path ]] || { echo "Missing required artifact: $path" >&2; exit 2; }
done
[[ ! -e $output ]] || { echo 'Fresh Episode-PIM destination already exists' >&2; exit 2; }

expected=(GPU-c4d9aceb-c274-fb29-e145-f6c2ef560bdc
          GPU-1b146bd7-326f-3563-b97e-24a26e13fa09
          GPU-ecaaa0cf-4454-f076-588a-c1667591a8d3
          GPU-045ff755-16e3-8b76-cd03-4cf0ae1996b6
          GPU-ef75e39b-bdd0-7f30-b1ae-199650d79299
          GPU-2adf85a6-fab7-9216-fb00-14b253a4cd3e
          GPU-3ccf761f-a31a-809b-90cd-cfdf9394277a
          GPU-323fc89e-78b5-e87e-91c3-f10569ac4e56)
visible=()
for physical in 0 1 2 3 4 5 6 7; do
  read -r actual used util < <(nvidia-smi -i "$physical" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
  [[ $actual == "${expected[$physical]}" && $used -lt 1024 && $util -le 5 ]] || {
    echo "GPU $physical is not the audited idle H100; refusing to interfere" >&2; exit 2;
  }
  visible+=("$actual")
done
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${visible[*]}")
mkdir -p "$run"

exec /usr/bin/python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$zeva_root/scripts/train_robotwin_episode_pim.py" \
  --dataset-root "$dataset" --cte-checkpoint "$cte" --cte-artifacts "$cte_artifacts" \
  --retrieval-checkpoint "$retrieval" --foundation-checkpoint "$foundation" \
  --parent-stage2-checkpoint "$parent" --handoff-root "$handoff" --save-dir "$output" \
  --steps 2000 --batch-size 32 --workers 2 --pim-lr 5e-5 --max-entries 64 \
  --exploratory-epoch40
