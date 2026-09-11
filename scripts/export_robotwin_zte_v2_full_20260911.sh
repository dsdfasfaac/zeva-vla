#!/usr/bin/env bash
set -Eeuo pipefail

# Durable, pinned full train95/validation export.  This is intentionally a
# four-process source shard export rather than distributed training: every
# process owns a deterministic slice of complete episodes and writes a CPU
# shard.  The merge and Stage 1.5 handoff run only after all shards pass.

repo_root=${ZEVA_REPO_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA}
handoff_root=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
dataset_root=${ROBOTWIN_DATASET_ROOT:-/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data}
goal_embeddings=${ROBOTWIN_GOAL_EMBEDDINGS:-/data1/dingxin/zeva-runs/robotwin-v5-h15-tasklang/pi05_task_embeddings.pt}
checkpoint=${ROBOTWIN_ZTE_CHECKPOINT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-phase-vector-mse-4096-20260911h-ddp-recovery/zte_v2_step_002048.pth}
output_dir=${ROBOTWIN_ZTE_ARTIFACT_DIR:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-step2048-20260911}
expected_checkpoint_sha256=${ROBOTWIN_ZTE_CHECKPOINT_SHA256:-d3d21c0acac9855a0a06311472d9ad773d914b57df311cf2648bafa9bc14bc5d}
selection_json=${ROBOTWIN_ZTE_SELECTION_JSON:-}
exporter=${repo_root}/scripts/export_robotwin_zte_v2_artifacts.py
retrieval=${repo_root}/scripts/train_robotwin_task_retrieval.py

batch_size=${ROBOTWIN_ZTE_EXPORT_BATCH_SIZE:-2}
num_workers=${ROBOTWIN_ZTE_EXPORT_WORKERS:-2}
log_every=${ROBOTWIN_ZTE_EXPORT_LOG_EVERY:-100}
world_size=4
read -r -a gpus <<< "${ROBOTWIN_ZTE_GPU_LIST:-2 5 6 7}"
if (( ${#gpus[@]} != world_size )); then
  echo "ROBOTWIN_ZTE_GPU_LIST must contain exactly ${world_size} GPU ids." >&2
  exit 2
fi

# The wrapper supplies the proven native Transformers5/TorchCodec runtime.
export ZEVA_PROCESSES=1
export ZEVA_RUNTIME_DEPS=${ZEVA_RUNTIME_DEPS:-/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1}
export NATIVE_TRANSFORMERS_RUNTIME=${NATIVE_TRANSFORMERS_RUNTIME:-/data1/dingxin/transformers5-runtime}
export PI05_PYTHON=${PI05_PYTHON:-python3}
export ROBOTWIN_HANDOFF=${handoff_root}
export PYTHONUNBUFFERED=1

mkdir -p "${output_dir}"
shopt -s nullglob dotglob
existing=("${output_dir}"/*)
if (( ${#existing[@]} != 0 )); then
  echo "Refusing non-empty artifact directory; this run is pinned and never overwrites: ${output_dir}" >&2
  exit 2
fi
mkdir -p "${output_dir}/logs"
exec > >(tee -a "${output_dir}/launcher.log") 2>&1

if [[ -n "${selection_json}" ]]; then
  if [[ ! -s "${selection_json}" ]]; then
    echo "Selected-checkpoint metadata is missing or empty: ${selection_json}" >&2
    exit 2
  fi
  cp -- "${selection_json}" "${output_dir}/checkpoint_selection.json"
fi

echo "export_start=$(date -Is)"
echo "repo=${repo_root}"
echo "checkpoint=${checkpoint}"
echo "checkpoint_sha256_expected=${expected_checkpoint_sha256}"
echo "output=${output_dir}"
echo "source_counts_expected=train:26150 validation:1350 tasks:50"

if [[ ! -f "${checkpoint}" || ! -f "${goal_embeddings}" || ! -f "${exporter}" ]]; then
  echo "Missing pinned input/exporter." >&2
  exit 2
fi
actual_checkpoint_sha256=$(sha256sum "${checkpoint}" | awk '{print $1}')
if [[ "${actual_checkpoint_sha256}" != "${expected_checkpoint_sha256}" ]]; then
  echo "Pinned checkpoint SHA mismatch: actual=${actual_checkpoint_sha256}" >&2
  exit 2
fi
echo "checkpoint_sha256_actual=${actual_checkpoint_sha256}"

# Keep the full source identity check in the Python exporter, but fail early
# if the checkpoint's goal provenance and selected goal file already disagree.
"${PI05_PYTHON}" - "${checkpoint}" "${goal_embeddings}" <<'PY'
import hashlib
import sys
import torch

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

checkpoint, goals = sys.argv[1:]
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
expected = payload.get("manifest", {}).get("goal_embeddings_sha256")
actual = sha256(goals)
if not expected:
    raise SystemExit("checkpoint has no manifest.goal_embeddings_sha256")
if expected != actual:
    raise SystemExit(f"goal embedding SHA mismatch: checkpoint={expected} selected={actual}")
print(f"goal_embeddings_sha256={actual}")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the pinned GPU export." >&2
  exit 2
fi
for gpu in "${gpus[@]}"; do
  state=$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  echo "gpu${gpu}_before=${state}"
done

declare -a pids=()
for rank in 0 1 2 3; do
  gpu=${gpus[$rank]}
  log="${output_dir}/logs/rank${rank}.log"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    cd "${repo_root}"
    bash scripts/run_robotwin_zte_v2.sh --run "${exporter}" \
      --handoff-root "${handoff_root}" \
      --dataset-root "${dataset_root}" \
      --goal-embeddings "${goal_embeddings}" \
      --checkpoint "${checkpoint}" \
      --output-dir "${output_dir}" \
      --batch-size "${batch_size}" \
      --num-workers "${num_workers}" \
      --log-every "${log_every}" \
      --rank "${rank}" \
      --world-size "${world_size}" \
      --device cuda
  ) >"${log}" 2>&1 &
  pids[$rank]=$!
  echo "${pids[$rank]}" > "${output_dir}/logs/rank${rank}.pid"
  echo "started rank=${rank} gpu=${gpu} pid=${pids[$rank]} log=${log}"
done

failed=0
for rank in 0 1 2 3; do
  if wait "${pids[$rank]}"; then
    echo "rank=${rank} status=success"
  else
    status=$?
    failed=1
    echo "rank=${rank} status=failed exit=${status}"
    tail -n 80 "${output_dir}/logs/rank${rank}.log" || true
  fi
done
if (( failed )); then
  echo "At least one shard failed; merge and retrieval are intentionally skipped." >&2
  exit 1
fi

echo "all_shards_complete=$(date -Is)"
cd "${repo_root}"
CUDA_VISIBLE_DEVICES="" bash scripts/run_robotwin_zte_v2.sh --run "${exporter}" \
  --handoff-root "${handoff_root}" \
  --dataset-root "${dataset_root}" \
  --goal-embeddings "${goal_embeddings}" \
  --checkpoint "${checkpoint}" \
  --output-dir "${output_dir}" \
  --batch-size "${batch_size}" \
  --num-workers "${num_workers}" \
  --log-every "${log_every}" \
  --rank 0 \
  --world-size "${world_size}" \
  --merge-shards \
  --device cpu >"${output_dir}/logs/merge.log" 2>&1

echo "merge_complete=$(date -Is)"

# Verify the merged artifact before handing it to Stage 1.5.  Retrieval is
# never allowed to consume a bounded/incomplete bank or a partial live cache.
"${PI05_PYTHON}" - "${output_dir}" "${checkpoint}" "${goal_embeddings}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
import torch

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

out, checkpoint, goals = map(Path, sys.argv[1:])
bank = torch.load(out / "train_causal_bank.pt", map_location="cpu", weights_only=False)
live = torch.load(out / "live_queries_h15.pt", map_location="cpu", weights_only=False)
expected_ckpt = sha256(checkpoint)
expected_goal = sha256(goals)
if bank.get("incomplete") or not bank.get("usable_for_training"):
    raise SystemExit("merged bank is marked incomplete/unusable")
if live.get("incomplete") or not live.get("usable_for_training"):
    raise SystemExit("merged live cache is marked incomplete/unusable")
if bank.get("manifest", {}).get("stage1_checkpoint_sha256") != expected_ckpt:
    raise SystemExit("bank checkpoint provenance mismatch")
if live.get("zte_checkpoint_sha256") != expected_ckpt:
    raise SystemExit("live checkpoint provenance mismatch")
if bank.get("manifest", {}).get("goal_embeddings_sha256") != expected_goal:
    raise SystemExit("bank goal provenance mismatch")
if live.get("goal_embeddings_sha256") != expected_goal:
    raise SystemExit("live goal provenance mismatch")
if tuple(bank["count"].shape) != (50, 32) or not torch.all(bank["count"].sum(dim=1) > 0):
    raise SystemExit("bank task/bin coverage is incomplete")
counts = live.get("source_record_counts", {})
exported = live.get("exported_record_counts", {})
if counts != {"train": 26150, "validation": 1350} or exported != counts:
    raise SystemExit(f"source coverage mismatch: counts={counts} exported={exported}")
for split, expected in counts.items():
    rows = live["splits"][split]
    ids = [int(row["record_index"]) for row in rows]
    if ids != list(range(expected)):
        raise SystemExit(f"{split} source record identity is not complete ordered coverage")
    for row in rows:
        phases = torch.as_tensor(row["phase_queries"])
        signals = torch.as_tensor(row["causal_signals"])
        if phases.shape[0] != len(row["decision_frames"]):
            raise SystemExit(f"{split} phase/live length mismatch at {row['record_index']}")
        if signals.shape[0] != phases.shape[0] - 1:
            raise SystemExit(f"{split} causal/live length mismatch at {row['record_index']}")
summary = {
    "schema": "zeva-robotwin-zte-v2-full-export-verification-v1",
    "checkpoint_sha256": expected_ckpt,
    "goal_embeddings_sha256": expected_goal,
    "train_records": counts["train"],
    "validation_records": counts["validation"],
    "tasks": len(bank["task_names"]),
    "usable_for_training": True,
}
(out / "verification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

echo "artifact_verification_complete=$(date -Is)"

retrieval_output="${output_dir}/task_retrieval.pth"
if [[ -e "${retrieval_output}" ]]; then
  echo "Refusing to overwrite existing retrieval output: ${retrieval_output}" >&2
  exit 2
fi
retrieval_gpu=${ROBOTWIN_ZTE_RETRIEVAL_GPU:-${gpus[0]}}
if ! printf '%s\n' "${gpus[@]}" | grep -Fxq "${retrieval_gpu}"; then
  echo "Retrieval GPU ${retrieval_gpu} is outside the export GPU set ${gpus[*]}." >&2
  exit 2
fi
retrieval_compute=$(nvidia-smi -i "${retrieval_gpu}" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
if [[ -n "${retrieval_compute}" ]]; then
  echo "Retrieval GPU ${retrieval_gpu} is occupied; refusing overlap:" >&2
  echo "${retrieval_compute}" >&2
  exit 1
fi
echo "stage1_5_start=$(date -Is) gpu=${retrieval_gpu}"
CUDA_VISIBLE_DEVICES="${retrieval_gpu}" cd "${repo_root}"
CUDA_VISIBLE_DEVICES="${retrieval_gpu}" bash scripts/run_robotwin_zte_v2.sh --run "${retrieval}" \
  --handoff-root "${handoff_root}" \
  --dataset-root "${dataset_root}" \
  --goal-embeddings "${goal_embeddings}" \
  --causal-bank "${output_dir}/train_causal_bank.pt" \
  --output "${retrieval_output}" \
  >"${output_dir}/logs/task_retrieval.log" 2>&1
echo "stage1_5_complete=$(date -Is)"
