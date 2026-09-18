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
preserve_weight=${BASE1000_ZTEV2_PRESERVE_WEIGHT:-8}
[[ "$preserve_weight" == 8 || "$preserve_weight" == 0 ]] || { echo "Only preregistered preserve weights 8/0 are supported" >&2; exit 2; }
mode=${1:-full}
base=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/fixed-anchor-pair-20260914/baseline/001000
stage1_root=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911
zte=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-phase-vector-mse-4096-20260911h-scheduler-repair-20260911i/zte_v2_step_004096.pth
cache=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/base1000-residual-probe-20260917/base_action_cache.pt
host=$(hostname -s)
dataset_identity_root=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang
replica_args=()
cache_only_args=()
case "$host" in
  aigc29)
    dataset=/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data
    gpu_index=0
    expected_uuid=GPU-dc037006-cc5d-131d-9b69-cae335dcf41f
    max_used_mib=1024
    ;;
  aigc31)
    # This cached-action path never decodes video.  The local EEF/Joint/stats
    # indices were copied from the fully audited aigc24 replica and rehashed;
    # source videos are intentionally absent from this host.
    dataset=/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data
    gpu_index=0
    expected_uuid=GPU-da0bafe1-add8-e332-399b-c2c50db124fe
    # A pre-existing idle 3.8 GiB CUDA reservation spans all eight cards on
    # this host.  Use one card only if >=60 GiB remains free and utilization
    # is <=5%; do not stop or reset the reserving process.
    max_used_mib=8192
    source_identity=$dataset_identity_root/dataset-identity-aigc29-stage2-resume-20260912.json
    replica_identity=$dataset_identity_root/dataset-identity-aigc24-stage2-resume-20260912.json
    [[ $(sha256sum "$source_identity" | awk '{print $1}') == 3ba76d89295564cabb735aecc47ddb55cb039fe21f4072138a14d1f02fb20666 ]] || exit 2
    [[ $(sha256sum "$replica_identity" | awk '{print $1}') == 73060be4ce82244631def725e8a05a1216aafd52550fa295221f79bc61f511d5 ]] || exit 2
    replica_args=(--dataset-identity-source-report "$source_identity" --dataset-identity-replica-report "$replica_identity")
    cache_only_args=(--cache-only-dataset-replica)
    ;;
  aigc28)
    # The action/Joint/stats indices were independently rehashed on this host
    # against the complete source/replica identity reports. Cached training
    # does not read the local source videos.
    dataset=/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data
    gpu_index=0
    expected_uuid=GPU-c4d9aceb-c274-fb29-e145-f6c2ef560bdc
    max_used_mib=1024
    source_identity=$dataset_identity_root/dataset-identity-aigc29-stage2-resume-20260912.json
    replica_identity=$dataset_identity_root/dataset-identity-aigc24-stage2-resume-20260912.json
    [[ $(sha256sum "$source_identity" | awk '{print $1}') == 3ba76d89295564cabb735aecc47ddb55cb039fe21f4072138a14d1f02fb20666 ]] || exit 2
    [[ $(sha256sum "$replica_identity" | awk '{print $1}') == 73060be4ce82244631def725e8a05a1216aafd52550fa295221f79bc61f511d5 ]] || exit 2
    replica_args=(--dataset-identity-source-report "$source_identity" --dataset-identity-replica-report "$replica_identity")
    cache_only_args=(--cache-only-dataset-replica)
    ;;
  *) echo "Only content-verified aigc28/aigc29/aigc31 are supported (aigc24 CUDA timeout)" >&2; exit 2 ;;
esac
[[ "$mode" == smoke || "$mode" == full ]] || { echo "Usage: $0 [smoke|full]" >&2; exit 2; }
for required in "$base/model.safetensors" "$zte" "$cache" "$dataset/adapter.json" \
  "$stage1_root/train_causal_bank.pt" "$stage1_root/live_queries_h15.pt" \
  "$stage1_root/task_retrieval.pth" "$zeva_root/configs/robotwin_zeva_advantage10.json"; do
  [[ -s "$required" ]] || { echo "Missing $required" >&2; exit 2; }
done
actual_uuid=$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader)
[[ "$actual_uuid" == "$expected_uuid" ]] || { echo "GPU UUID changed" >&2; exit 2; }
gpu_row=$(nvidia-smi -i "$gpu_index" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
IFS=, read -r used_mib free_mib utilization <<< "$gpu_row"
used_mib=${used_mib//[[:space:]]/}
free_mib=${free_mib//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
(( used_mib < max_used_mib && free_mib > 60000 && utilization <= 5 )) || { echo "Selected GPU is busy" >&2; exit 2; }

output=$run_root/$mode
[[ ! -e "$output/manifest.json" ]] || { echo "Refusing to overwrite $output" >&2; exit 2; }
mkdir -p "$output"
exec > "$output/launcher.log" 2>&1
steps=500
save_freq=250
extra_args=()
if [[ "$mode" == smoke ]]; then
  steps=2
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

"$python_bin" -u "$zeva_root/scripts/train_robotwin_stage2.py" \
  --handoff-root "$handoff" \
  --foundation-checkpoint "$handoff/checkpoint/pretrained_model-best-v1" \
  --initial-stage2-checkpoint "$base" \
  --goal-embedding-checkpoint "$handoff/checkpoint/pretrained_model" \
  --training-variant output_residual \
  --dataset-root "$dataset" \
  "${replica_args[@]}" \
  "${cache_only_args[@]}" \
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
  --prior-loss-weight 0 --preserve-loss-weight "$preserve_weight" \
  --paired-improvement-margin 0 --gate-regularization-weight 0.0001 \
  --residual-bound 0.2 --residual-regression-weight 0.25 \
  --residual-trust-region-weight 0.1 --residual-trust-region-radius 0.05 \
  --prior-residual-dropout-probability 0 \
  --learning-rate 5e-5 --warmup-steps 50 \
  --eval-batches 1000000 --log-freq 10 --no-compile-model \
  "${extra_args[@]}"

printf 'completed %s\n' "$(date --iso-8601=seconds)" > "$output/COMPLETED"
