#!/usr/bin/env bash
# Explicit, isolated epoch40 exploratory branch. The fixed epoch80 launcher is unchanged.
set -euo pipefail
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-}
[[ $(hostname -s) == aigc28 ]] || { echo 'This audited run is restricted to aigc28' >&2; exit 2; }
run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-behavior-effect-20260918
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
dataset=/data1/dingxin/robotwin-lerobot-sidney-eef16-v1/data
cte=$run/stage1/cte_epoch_040.pth
artifact=$run/cte-epoch040-exploratory-artifacts.pth
retrieval=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911/task_retrieval.pth
foundation=$handoff/checkpoint/pretrained_model-best-v1
audit=$run/prestage2-epoch040-exploratory-audit.json
stage2=$run/stage2-fullpi-pbd-effect-epoch040-exploratory
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

[[ -f $cte ]] || { echo 'Actual epoch40 checkpoint is absent' >&2; exit 2; }
case "$mode" in
  export|stage2)
    uuids=(GPU-1b146bd7-326f-3563-b97e-24a26e13fa09
           GPU-ecaaa0cf-4454-f076-588a-c1667591a8d3
           GPU-045ff755-16e3-8b76-cd03-4cf0ae1996b6
           GPU-ef75e39b-bdd0-7f30-b1ae-199650d79299)
    count=4
    [[ "$mode" == export ]] && count=1
    visible=()
    for ((i=0;i<count;i++)); do
      physical=$((i+1))
      read -r actual used util < <(nvidia-smi -i "$physical" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
      [[ "$actual" == "${uuids[$i]}" && "$used" -lt 1024 && "$util" -le 5 ]] || {
        echo "GPU $physical is not the audited idle H100; do not interfere with another job" >&2; exit 2;
      }
      visible+=("${uuids[$i]}")
    done
    export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${visible[*]}")
    ;;
  preflight) ;;
  *) echo 'Usage: export | preflight | stage2' >&2; exit 2 ;;
esac

if [[ "$mode" == export ]]; then
  [[ ! -e $artifact && ! -e ${artifact%.pth}.tmp ]] || { echo 'Exploratory export exists; inspect before retrying' >&2; exit 2; }
  exec /usr/bin/python3 -u "$zeva_root/scripts/export_robotwin_behavior_effect.py" \
    --checkpoint "$cte" --dataset-root "$dataset" --output "$artifact" \
    --task-subset "$zeva_root/configs/robotwin_zeva_advantage10.json" --handoff-root "$handoff" \
    --exploratory-epoch40
fi
[[ -f $artifact ]] || { echo 'Complete exploratory CTE artifact is absent' >&2; exit 2; }
if [[ "$mode" == preflight ]]; then
  [[ ! -e $audit ]] || { echo 'Exploratory audit already exists' >&2; exit 2; }
  exec /usr/bin/python3 -u "$zeva_root/scripts/audit_robotwin_behavior_effect_artifacts.py" \
    --checkpoint "$cte" --artifacts "$artifact" --retrieval "$retrieval" \
    --dataset-root "$dataset" --foundation "$foundation" --handoff "$handoff" \
    --tasks "$zeva_root/configs/robotwin_zeva_advantage10.json" --output "$audit" \
    --exploratory-epoch40
fi
[[ -f $audit ]] || { echo 'Independent epoch40 artifact audit is absent' >&2; exit 2; }
[[ ! -e $stage2 ]] || { echo 'Fresh exploratory Stage2 destination already exists' >&2; exit 2; }
/usr/bin/python3 - "$audit" "$cte" "$artifact" "$retrieval" "$foundation/model.safetensors" \
  "$dataset/adapter.json" "$handoff/reference/mean-std-eef16-h50-stage1grip-train95-v2.json" \
  "$zeva_root/scripts/audit_robotwin_behavior_effect_artifacts.py" <<'PY'
import hashlib, json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
if (report.get("schema") != "zeva-behavior-effect-prestage2-audit-v1"
        or report.get("status") != "PASS" or report.get("stage1_epoch") != 40
        or report.get("exploratory_epoch40") is not True
        or report.get("formal_promotion_eligible") is not False):
    raise SystemExit("Exploratory preflight report did not pass")
for label, arg in zip(("checkpoint", "artifact", "retrieval", "foundation", "adapter", "statistics", "audit_script"), sys.argv[2:]):
    digest = hashlib.sha256()
    with open(arg, "rb") as stream:
        for block in iter(lambda:stream.read(4 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != report["sha256"][label]:
        raise SystemExit(f"Exploratory preflight {label} SHA changed")
PY
exec /usr/bin/python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$zeva_root/scripts/train_robotwin_behavior_effect_policy.py" \
  --dataset-root "$dataset" --cte-checkpoint "$cte" --artifacts "$artifact" \
  --retrieval-checkpoint "$retrieval" --foundation-checkpoint "$foundation" \
  --handoff-root "$handoff" --save-dir "$stage2" \
  --steps 5000 --batch-size 8 --accumulation 8 --workers 2 --exploratory-epoch40
