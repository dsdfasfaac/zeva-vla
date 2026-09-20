#!/usr/bin/env bash
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ $(hostname -s) == aigc28 ]] || { echo 'Episode-PIM validation is restricted to aigc28' >&2; exit 2; }

run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-episode-pim-20260920
parent_run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-behavior-effect-20260918
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
dataset=/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data
output=$run/validation5-episode-pim-step2000-8shard
checkpoint=$run/stage2-cte-bit-episode-pim-eap-fixed2000-8gpu/002000
parent=$parent_run/stage2-fullpi-pbd-effect-epoch040-exploratory/005000
cte=$parent_run/stage1/cte_epoch_040.pth
cte_artifacts=$parent_run/cte-epoch040-exploratory-artifacts.pth
retrieval=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911/task_retrieval.pth
foundation=$handoff/checkpoint/pretrained_model-best-v1
expected=$parent_run/validation5-expected-decisions.json

runtime=$handoff/runtime
overlay=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
native_transformers=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
compiled_deps=/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1
shared_deps=/data1/dingxin/zeva-runtime-deps
export PYTHONPATH="$overlay:$native_transformers:$shared_deps:$compiled_deps:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$runtime/lerobot-main-py311-v1/src:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export LD_LIBRARY_PATH="$overlay/torch/lib:$overlay/nvidia/cuda_runtime/lib:$overlay/nvidia/cublas/lib:$overlay/nvidia/cudnn/lib:$overlay/nvidia/nccl/lib:$overlay/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCH_HOME=/mnt/100T/users/dingxin/VLA/runtime/zeva-torch-cache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for path in "$checkpoint/COMPLETE" "$parent/COMPLETE" "$cte" "$cte_artifacts" \
            "$retrieval" "$foundation/model.safetensors" "$expected"; do
  [[ -f $path ]] || { echo "Missing validation input: $path" >&2; exit 2; }
done
[[ ! -e $output ]] || { echo "Refusing to overwrite validation output: $output" >&2; exit 2; }

expected_gpus=(GPU-c4d9aceb-c274-fb29-e145-f6c2ef560bdc
               GPU-1b146bd7-326f-3563-b97e-24a26e13fa09
               GPU-ecaaa0cf-4454-f076-588a-c1667591a8d3
               GPU-045ff755-16e3-8b76-cd03-4cf0ae1996b6
               GPU-ef75e39b-bdd0-7f30-b1ae-199650d79299
               GPU-2adf85a6-fab7-9216-fb00-14b253a4cd3e
               GPU-3ccf761f-a31a-809b-90cd-cfdf9394277a
               GPU-323fc89e-78b5-e87e-91c3-f10569ac4e56)
for gpu in "${expected_gpus[@]}"; do
  read -r used util < <(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
  [[ $used -lt 1024 && $util -le 5 ]] || { echo "GPU $gpu is not safely idle" >&2; exit 2; }
done

mkdir "$output"
common=("$zeva_root/scripts/report_robotwin_episode_pim_validation.py"
  --dataset-root "$dataset" --cte-checkpoint "$cte" --cte-artifacts "$cte_artifacts"
  --retrieval-checkpoint "$retrieval" --foundation-checkpoint "$foundation"
  --parent-checkpoint "$parent" --checkpoint "$checkpoint" --expected-decisions "$expected"
  --output-dir "$output" --tasks "$zeva_root/configs/robotwin_zeva_advantage10.json"
  --handoff "$handoff" --num-shards 8)

pids=()
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=${expected_gpus[$rank]} /usr/bin/python3 "${common[@]}" \
    --shard-index "$rank" >"$output/shard-$rank.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" >"$output/pids.txt"
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || { echo 'One or more Episode-PIM validation shards failed' >&2; exit 1; }
/usr/bin/python3 "${common[@]}" --shard-index 0 --merge >"$output/merge.log" 2>&1
printf 'complete frozen Episode-PIM validation5 report\n' >"$output/COMPLETE"
