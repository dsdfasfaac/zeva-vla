#!/usr/bin/env bash
set -euo pipefail

# Fresh, two-arm Stage 2 mechanism check for the initial dual-residual gate.
#
# Both arms are training_variant=zeva, start from the same completed
# Base/004500 weights, load that same checkpoint as the immutable action-path
# teacher, and create a fresh optimizer.  The only arm-level change is
# initial_residual_gate_probability (0.01 versus 0.10).  This script is kept
# separate from the historical fixed-anchor launcher and never mutates the
# native Transformers/runtime symlinks.

mode=${1:-both}
case "$mode" in
  gate001|gate010|both) ;;
  nll_detached|nll_detached_full|gradient_route_full|gradient_route_smoke|h15_route_full|h15_route_smoke) ;;
  *)
    echo "usage: $0 [gate001|gate010|both|nll_detached|nll_detached_full|gradient_route_full|gradient_route_smoke|h15_route_full|h15_route_smoke]" >&2
    exit 2
    ;;
esac

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
contract_config="$zeva_root/configs/robotwin_ztev2_gate_mechanism_20260915.json"
if [[ "$mode" == nll_detached ]]; then
  contract_config="$zeva_root/configs/robotwin_ztev2_nll_routing_20260915.json"
elif [[ "$mode" == nll_detached_full ]]; then
  contract_config="$zeva_root/configs/robotwin_ztev2_nll_routing_full_20260915.json"
elif [[ "$mode" == gradient_route_full || "$mode" == gradient_route_smoke ]]; then
  contract_config="$zeva_root/configs/robotwin_ztev2_gradient_route_full_20260916.json"
elif [[ "$mode" == h15_route_full || "$mode" == h15_route_smoke ]]; then
  contract_config="$zeva_root/configs/robotwin_ztev2_h15_route_20260916.json"
fi

handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
runtime=${ROBOTWIN_RUNTIME:-$handoff/runtime}
python_bin=${PI05_PYTHON:-python3}
native_transformers=${NATIVE_TRANSFORMERS_RUNTIME:-/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912}
shared_deps=${ZEVA_RUNTIME_DEPS:-/data1/dingxin/zeva-runtime-deps}
compiled_deps=${ZEVA_COMPILED_DEPS:-/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1}
model_dependency_overlay=${MODEL_DEPENDENCY_OVERLAY:-}
model_ld_library_path=${MODEL_LD_LIBRARY_PATH:-}

foundation=${ROBOTWIN_FOUNDATION:-$handoff/checkpoint/pretrained_model-best-v1}
foundation_sha256=${ROBOTWIN_FOUNDATION_SHA256:-7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe}
stage1_language=${ROBOTWIN_STAGE1_LANGUAGE:-$handoff/checkpoint/pretrained_model}
dataset_root=${ROBOTWIN_DATASET_ROOT:-/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data}
task_subset=${ROBOTWIN_TASK_SUBSET:-$zeva_root/configs/robotwin_zeva_advantage10.json}
base_checkpoint=${ROBOTWIN_TRAINED_BASE:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-ztev2-schedulerfix-pair-20260911/baseline/004500}
base_sha256=${ROBOTWIN_TRAINED_BASE_SHA256:-2f106633403e5f2146bf7cd4f56858cbfdb856e9b2966d1c724a79ac4948c84f}
base_manifest_sha256=${ROBOTWIN_TRAINED_BASE_MANIFEST_SHA256:-b768c6ff917c94491970caae8a9f814f139b53a66cf7dc0fb9393a3333473750}
if [[ "$mode" == h15_route_full || "$mode" == h15_route_smoke ]]; then
  base_checkpoint=${ROBOTWIN_TRAINED_BASE:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/fixed-anchor-pair-20260914/baseline/001000}
  base_sha256=${ROBOTWIN_TRAINED_BASE_SHA256:-bcf1d4f5f3e77926b8b7f798bb983378ad096e3892eed1a66058c5d98ed9bc17}
  base_manifest_sha256=${ROBOTWIN_TRAINED_BASE_MANIFEST_SHA256:-630d30646bf5a75d6d7207865b7bd572a784d370dec5b8037ebdbc6e9a26c3d0}
fi
stage1_root=${ROBOTWIN_STAGE1_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911}
zte_checkpoint=${ROBOTWIN_ZTE_CHECKPOINT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-phase-vector-mse-4096-20260911h-scheduler-repair-20260911i/zte_v2_step_004096.pth}
causal_bank=${ROBOTWIN_CAUSAL_BANK:-$stage1_root/train_causal_bank.pt}
live_queries=${ROBOTWIN_LIVE_QUERIES:-$stage1_root/live_queries_h15.pt}
task_retrieval=${ROBOTWIN_TASK_RETRIEVAL:-$stage1_root/task_retrieval.pth}

dataset_adapter_sha256=${ROBOTWIN_DATASET_ADAPTER_SHA256:-8ac54abcec7704b0111b7c28be3fb3a18e27e0ebe8e3dcf8ddff948b36100f8f}
zte_sha256=${ROBOTWIN_ZTE_SHA256:-ce9402981b8e2f1ce597e2cb150b795ae246f84fb81b3bb72fae023a3decf899}
causal_bank_sha256=${ROBOTWIN_CAUSAL_BANK_SHA256:-33b11b41992d07890a798602cb8cadebc0f798b8959adfc3b8cf5b660bf14ad9}
live_queries_sha256=${ROBOTWIN_LIVE_QUERIES_SHA256:-c39e3c6565c438b9496133833a7288a08a81a7985fc1723bf8a39d6ad2d47de6}
task_retrieval_sha256=${ROBOTWIN_TASK_RETRIEVAL_SHA256:-c54b3275af4b15999a5d849f476c19ff3d01c76150a487679e9bbe61928798b7}

run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/gate-mechanism-20260915}
if [[ "$mode" == nll_detached ]]; then
  run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/nll-routing-20260915}
elif [[ "$mode" == nll_detached_full ]]; then
  run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/nll-routing-full-20260915}
elif [[ "$mode" == gradient_route_full ]]; then
  run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/gradient-route-full-20260916}
elif [[ "$mode" == gradient_route_smoke ]]; then
  run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/gradient-route-smoke-20260916}
elif [[ "$mode" == h15_route_full || "$mode" == h15_route_smoke ]]; then
  run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/h15-route-20260916}
fi

# The mechanism check is deliberately short, but its optimization/data
# contract is otherwise the formal fixed-anchor contract.
steps=100
warmup_steps=10
save_freq=100
if [[ "$mode" == nll_detached_full || "$mode" == gradient_route_full ]]; then
  steps=1000
  warmup_steps=100
  save_freq=250
elif [[ "$mode" == gradient_route_smoke ]]; then
  steps=1
  warmup_steps=1
  save_freq=1
elif [[ "$mode" == h15_route_full ]]; then
  steps=1000
  warmup_steps=100
  save_freq=500
elif [[ "$mode" == h15_route_smoke ]]; then
  steps=1
  warmup_steps=1
  save_freq=1
fi
batch_size=16
gradient_accumulation_steps=4
num_processes=4
num_workers=${NUM_WORKERS:-4}
eval_batches=1000000
action_expert_learning_rate=5e-6
if [[ "$mode" == h15_route_full || "$mode" == h15_route_smoke ]]; then
  action_expert_learning_rate=0
fi
zeva_learning_rate=5e-5
prior_loss_weight=0.01
preserve_loss_weight=1.0
paired_improvement_margin=0.0
baseline_preserve_interval=4
gate_regularization_weight=0.001
prior_residual_dropout_probability=0.4
phase_noise_std=0.02
memory_dropout=0.1
prior_injection_horizon=50
compile_mode=default
seed=1000

gpu_list=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
gpu_memory_limit_mib=${GPU_MEMORY_LIMIT_MIB:-1024}
gpu_utilization_limit_pct=${GPU_UTILIZATION_LIMIT_PCT:-5}

case "$mode" in
  gate001) arms=(gate001) ;;
  gate010) arms=(gate010) ;;
  nll_detached) arms=(nll_detached) ;;
  nll_detached_full) arms=(nll_detached_full) ;;
  gradient_route_full) arms=(gradient_route_full) ;;
  gradient_route_smoke) arms=(gradient_route_smoke) ;;
  h15_route_full) arms=(h15_route_full) ;;
  h15_route_smoke) arms=(h15_route_smoke) ;;
  both) arms=(gate001 gate010) ;;
esac

die() {
  echo "gate-mechanism preflight: $*" >&2
  exit 3
}

require_file() {
  local path=$1
  [[ -s "$path" ]] || die "missing or empty required file: $path"
}

require_dir() {
  local path=$1
  [[ -d "$path" ]] || die "missing required directory: $path"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

safe_absolute_path() {
  [[ "$1" =~ ^/[A-Za-z0-9_./:-]+$ ]]
}

[[ -f "$contract_config" ]] || die "missing launcher contract: $contract_config"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required for lineage checks"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required for the idle-GPU preflight"
"$python_bin" --version >/dev/null 2>&1 || die "PI05_PYTHON is not executable: $python_bin"

require_dir "$runtime"
require_dir "$native_transformers"
require_dir "$shared_deps"
require_dir "$compiled_deps"
if [[ -n "$model_dependency_overlay" ]]; then
  safe_absolute_path "$model_dependency_overlay" || die "MODEL_DEPENDENCY_OVERLAY must be a safe absolute path"
  require_dir "$model_dependency_overlay"
fi
if [[ -n "$model_ld_library_path" ]]; then
  safe_absolute_path "$model_ld_library_path" || die "MODEL_LD_LIBRARY_PATH must contain safe absolute paths"
  export LD_LIBRARY_PATH="$model_ld_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

system_site=$(
  "$python_bin" -c 'import site; print(site.getsitepackages()[0])'
)
export EGOSCALE_LEROBOT_SOURCE="$runtime/lerobot-main-py311-v1/src"
pythonpath_prefix="$native_transformers:$shared_deps:$compiled_deps:$system_site"
if [[ -n "$model_dependency_overlay" ]]; then
  pythonpath_prefix="$model_dependency_overlay:$pythonpath_prefix"
fi
export PYTHONPATH="$pythonpath_prefix:$runtime/h100-extra-deps:$runtime/lerobot-overlay-v2:$EGOSCALE_LEROBOT_SOURCE:$runtime/src:$zeva_root/src:$zeva_root:$runtime/lerobot-main-deps-py311-v1"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-32}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

check_tmpdir_resource_sharing() {
  local supplied_tmpdir=${TMPDIR:-<unset>}
  if ! "$python_bin" - <<'PY'
import multiprocessing.resource_sharer as resource_sharer
import multiprocessing.util as multiprocessing_util
import os
import sys
import tempfile


read_fd = write_fd = shared_fd = None
try:
    supplied_tmpdir = os.environ.get("TMPDIR")
    actual_tmpdir = tempfile.gettempdir()
    if supplied_tmpdir is not None and os.path.realpath(actual_tmpdir) != os.path.realpath(supplied_tmpdir):
        raise RuntimeError(
            f"Python resolved TMPDIR={actual_tmpdir!r} instead of supplied "
            f"TMPDIR={supplied_tmpdir!r}"
        )

    # Linux may prefer abstract sockets, which do not exercise the filesystem
    # AF_UNIX path used by runtimes where the long-TMPDIR failure occurs.
    # This assignment is confined to this probe subprocess; training workers
    # inherit the caller's TMPDIR and Python configuration unchanged.
    multiprocessing_util.abstract_sockets_supported = False

    read_fd, write_fd = os.pipe()
    shared_fd = resource_sharer.DupFd(write_fd).detach()
    payload = b"zeva-tmpdir-resource-sharer-preflight"
    os.write(shared_fd, payload)
    received = os.read(read_fd, len(payload))
    if received != payload:
        raise RuntimeError(f"pipe payload mismatch: received {received!r}")
except BaseException as error:
    tmpdir = os.environ.get("TMPDIR") or tempfile.gettempdir()
    print(
        "TMPDIR resource-sharing preflight failed "
        f"(TMPDIR={tmpdir!r}): {type(error).__name__}: {error}",
        file=sys.stderr,
    )
    raise SystemExit(1)
finally:
    for fd in (shared_fd, write_fd, read_fd):
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
    resource_sharer.stop()
PY
  then
    die "TMPDIR=$supplied_tmpdir cannot support multiprocessing.resource_sharer.DupFd; choose a shorter existing directory"
  fi
}

check_tmpdir_resource_sharing

require_file "$foundation/model.safetensors"
require_file "$base_checkpoint/model.safetensors"
require_file "$base_checkpoint/training_state.pt"
require_file "$base_checkpoint/../manifest.json"
require_file "$dataset_root/adapter.json"
require_file "$task_subset"
require_file "$stage1_language/model.safetensors"
require_file "$zte_checkpoint"
require_file "$causal_bank"
require_file "$live_queries"
require_file "$task_retrieval"

actual_foundation_sha256=$(sha256_file "$foundation/model.safetensors")
[[ "$actual_foundation_sha256" == "$foundation_sha256" ]] || die \
  "unexpected best-v1 foundation SHA256: $actual_foundation_sha256 (expected $foundation_sha256)"
actual_base_sha256=$(sha256_file "$base_checkpoint/model.safetensors")
[[ "$actual_base_sha256" == "$base_sha256" ]] || die \
  "unexpected Base/004500 model SHA256: $actual_base_sha256 (expected $base_sha256)"
actual_base_manifest_sha256=$(sha256_file "$base_checkpoint/../manifest.json")
[[ "$actual_base_manifest_sha256" == "$base_manifest_sha256" ]] || die \
  "unexpected Base source manifest SHA256: $actual_base_manifest_sha256 (expected $base_manifest_sha256)"
actual_dataset_adapter_sha256=$(sha256_file "$dataset_root/adapter.json")
[[ "$actual_dataset_adapter_sha256" == "$dataset_adapter_sha256" ]] || die \
  "dataset adapter SHA256 differs from the frozen bank lineage: $actual_dataset_adapter_sha256"
actual_zte_sha256=$(sha256_file "$zte_checkpoint")
[[ "$actual_zte_sha256" == "$zte_sha256" ]] || die \
  "ZTE checkpoint SHA256 differs from the frozen bank lineage: $actual_zte_sha256"
actual_causal_bank_sha256=$(sha256_file "$causal_bank")
[[ "$actual_causal_bank_sha256" == "$causal_bank_sha256" ]] || die \
  "causal bank SHA256 differs from the frozen Stage1 lineage: $actual_causal_bank_sha256"
actual_live_queries_sha256=$(sha256_file "$live_queries")
[[ "$actual_live_queries_sha256" == "$live_queries_sha256" ]] || die \
  "live-query SHA256 differs from the frozen Stage1 lineage: $actual_live_queries_sha256"
actual_task_retrieval_sha256=$(sha256_file "$task_retrieval")
[[ "$actual_task_retrieval_sha256" == "$task_retrieval_sha256" ]] || die \
  "task-retrieval SHA256 differs from the frozen Stage1 lineage: $actual_task_retrieval_sha256"

# The selected 004500 artifact must be the ordinary prior Base, not a ZeVA
# checkpoint.  Its manifest also pins the untouched foundation identity.
"$python_bin" - "$base_checkpoint/../manifest.json" "$base_checkpoint" "$foundation_sha256" "$mode" <<'PY'
import json
from pathlib import Path
import sys

manifest_path, checkpoint_path, expected_foundation, arm = sys.argv[1:]
payload = json.loads(Path(manifest_path).read_text())
if payload.get("training_variant") != "baseline":
    raise SystemExit("Base/004500 source manifest is not training_variant=baseline")
train_args = payload.get("train_args")
if not isinstance(train_args, dict) or train_args.get("training_variant") != "baseline":
    raise SystemExit("Base/004500 train_args do not declare the ordinary baseline")
identity = payload.get("foundation_identity")
if not isinstance(identity, dict) or identity.get("model_sha256") != expected_foundation:
    raise SystemExit("Base/004500 manifest foundation SHA256 does not match best-v1")
expected_step = "001000" if arm in ("h15_route_full", "h15_route_smoke") else "004500"
if Path(checkpoint_path).name != expected_step:
    raise SystemExit(f"selected Base checkpoint directory must be named {expected_step}")
PY

# This experiment is intentionally fresh.  Neither a resume state nor an old
# ZeVA adapter may enter either branch through the environment.
[[ -z "${RESUME_CHECKPOINT:-}" ]] || die "RESUME_CHECKPOINT is forbidden; this is a fresh optimizer"
[[ -z "${ZEVA_ADAPTER_CHECKPOINT:-}" ]] || die "ZEVA_ADAPTER_CHECKPOINT is forbidden; the adapter must be fresh"

IFS=',' read -r -a gpu_ids <<< "$gpu_list"
[[ "${#gpu_ids[@]}" -eq "$num_processes" ]] || die \
  "CUDA_VISIBLE_DEVICES=$gpu_list must list exactly $num_processes GPUs"
[[ "$gpu_memory_limit_mib" =~ ^[0-9]+$ ]] || die "GPU_MEMORY_LIMIT_MIB must be an integer"
[[ "$gpu_utilization_limit_pct" =~ ^[0-9]+$ ]] || die "GPU_UTILIZATION_LIMIT_PCT must be an integer"
# Resolve nvidia-smi indices to UUIDs before CUDA import. A failed physical GPU
# can make CUDA ordinal numbering differ from nvidia-smi numbering.
resolved_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ || "$gpu_id" =~ ^GPU-[0-9a-fA-F-]{36}$ ]] || die "invalid GPU selector: $gpu_id"
  gpu_uuid=$(nvidia-smi -i "$gpu_id" --query-gpu=uuid --format=csv,noheader)
  [[ "$gpu_uuid" =~ ^GPU-[0-9a-fA-F-]{36}$ ]] || die "could not resolve a single GPU UUID: $gpu_id"
  for resolved_gpu_id in ${resolved_gpu_ids[@]+"${resolved_gpu_ids[@]}"}; do
    [[ "$resolved_gpu_id" != "$gpu_uuid" ]] || die "duplicate GPU UUID: $gpu_uuid"
  done
  resolved_gpu_ids+=("$gpu_uuid")
done
gpu_ids=("${resolved_gpu_ids[@]}")
gpu_list=$(IFS=,; echo "${gpu_ids[*]}")
export CUDA_VISIBLE_DEVICES="$gpu_list"

torch_version=$(
  "$python_bin" -c 'import torch; print(torch.__version__.split("+")[0])'
)
[[ "$torch_version" == "2.7.1" ]] || die "verified gate mechanism runtime requires Torch2.7.1, got $torch_version"

check_idle_gpus() {
  local gpu_id row memory_used utilization
  for gpu_id in "${gpu_ids[@]}"; do
    row=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    memory_used=$(awk -F',' '{gsub(/[[:space:]]/, "", $1); print $1}' <<< "$row")
    utilization=$(awk -F',' '{gsub(/[[:space:]]/, "", $2); print $2}' <<< "$row")
    [[ "$memory_used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || die \
      "could not parse nvidia-smi row for GPU $gpu_id: $row"
    (( memory_used <= gpu_memory_limit_mib )) || die \
      "GPU $gpu_id has ${memory_used}MiB in use (limit ${gpu_memory_limit_mib}MiB)"
    (( utilization <= gpu_utilization_limit_pct )) || die \
      "GPU $gpu_id has ${utilization}% utilization (limit ${gpu_utilization_limit_pct}%)"
  done
}

check_port_free() {
  local port=$1
  [[ "$port" =~ ^[0-9]+$ ]] || die "DDP port must be numeric: $port"
  if command -v ss >/dev/null 2>&1; then
    if ! ss -ltnH | awk -v expected_port="$port" '
      {
        endpoint=$4
        sub(/^.*:/, "", endpoint)
        if (endpoint == expected_port) found=1
      }
      END { exit found ? 1 : 0 }
    '; then
      die "DDP port is already listening: $port"
    fi
  else
    "$python_bin" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", port))
finally:
    sock.close()
PY
  fi
}

port_for_arm() {
  case "$1" in
    gate001) printf '%s\n' "${GATE001_MAIN_PROCESS_PORT:-29614}" ;;
    gate010) printf '%s\n' "${GATE010_MAIN_PROCESS_PORT:-29615}" ;;
    nll_detached) printf '%s\n' "${NLL_ROUTING_MAIN_PROCESS_PORT:-29616}" ;;
    nll_detached_full) printf '%s\n' "${NLL_FULL_MAIN_PROCESS_PORT:-29617}" ;;
    gradient_route_full) printf '%s\n' "${GRADIENT_ROUTE_MAIN_PROCESS_PORT:-29618}" ;;
    gradient_route_smoke) printf '%s\n' "${GRADIENT_SMOKE_MAIN_PROCESS_PORT:-29619}" ;;
    h15_route_full) printf '%s\n' "${H15_ROUTE_MAIN_PROCESS_PORT:-29620}" ;;
    h15_route_smoke) printf '%s\n' "${H15_SMOKE_MAIN_PROCESS_PORT:-29621}" ;;
    *) die "unknown gate arm: $1" ;;
  esac
}

gate_probability_for_arm() {
  case "$1" in
    gate001) printf '%s\n' "0.01" ;;
    gate010) printf '%s\n' "0.10" ;;
    nll_detached) printf '%s\n' "0.01" ;;
    nll_detached_full) printf '%s\n' "0.01" ;;
    gradient_route_full) printf '%s\n' "0.01" ;;
    gradient_route_smoke) printf '%s\n' "0.01" ;;
    h15_route_full|h15_route_smoke) printf '%s\n' "0.01" ;;
    *) die "unknown gate arm: $1" ;;
  esac
}

for arm in "${arms[@]}"; do
  output="$run_root/$arm"
  compile_cache="$run_root/_compile-cache/$arm"
  [[ ! -e "$output" ]] || die "refusing to overwrite existing output: $output"
  [[ ! -e "$compile_cache" ]] || die "refusing to reuse existing compile cache: $compile_cache"
done
[[ ! -e "$run_root/COMPLETE" ]] || die "refusing to overwrite an existing pair marker: $run_root/COMPLETE"

write_launcher_manifest() {
  local arm=$1
  local output=$2
  local main_port=$3
  local gate_probability=$4
  local launcher_manifest="$output/launcher_manifest.json"
  "$python_bin" - "$launcher_manifest" "$contract_config" "$zeva_root/scripts/train_robotwin_ztev2_gate_mechanism.sh" \
    "$arm" "$gate_probability" "$main_port" "$run_root" "$handoff" "$runtime" \
    "$native_transformers" "$shared_deps" "$compiled_deps" "$model_dependency_overlay" \
    "$model_ld_library_path" "$foundation" "$foundation_sha256" "$base_checkpoint" "$base_sha256" \
    "$base_manifest_sha256" "$stage1_language" "$dataset_root" "$task_subset" "$zte_checkpoint" \
    "$zte_sha256" "$causal_bank" "$causal_bank_sha256" "$live_queries" "$live_queries_sha256" \
    "$task_retrieval" "$task_retrieval_sha256" "$dataset_adapter_sha256" "$steps" "$warmup_steps" \
    "$save_freq" "$batch_size" "$gradient_accumulation_steps" "$num_processes" "$gpu_list" "$seed" \
    "$action_expert_learning_rate" <<'PY'

import hashlib
import json
import importlib.util
import os
from pathlib import Path
import sys

(
    output, contract, launcher, arm, gate_probability, port, run_root, handoff, runtime,
    native_transformers, shared_deps, compiled_deps, dependency_overlay, ld_library_path,
    foundation, foundation_sha, base_checkpoint, base_sha, base_manifest_sha, stage1_language,
    dataset_root, task_subset, zte, zte_sha, bank, bank_sha, live_queries, live_queries_sha,
    retrieval, retrieval_sha, adapter_sha, steps, warmup_steps, save_freq, batch_size,
    accumulation, processes, gpu_list, seed, action_expert_learning_rate,
) = sys.argv[1:]

def digest(path: str) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def module_file(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin in (None, "built-in"):
        return None
    return str(Path(spec.origin).resolve())

payload = {
    "schema": "zeva-robotwin-stage2-gate-mechanism-launch-v1",
    "contract_config": {"path": str(Path(contract).resolve()), "sha256": digest(contract)},
    "launcher": {"path": str(Path(launcher).resolve()), "sha256": digest(launcher)},
    "arm": arm,
    "initial_residual_gate_probability": float(gate_probability),
    "training_variant": "zeva",
    "not_a_resume": True,
    "optimizer_reset": True,
    "not_a_matched_newly_trained_base_comparison": True,
    "ddp": {"num_processes": int(processes), "main_process_port": int(port), "gpu_list": gpu_list},
    "training": {
        "steps": int(steps), "warmup_steps": int(warmup_steps), "save_freq": int(save_freq),
        "batch_size_per_gpu": int(batch_size), "gradient_accumulation_steps": int(accumulation),
        "global_batch_size": int(batch_size) * int(accumulation) * int(processes),
        "action_expert_learning_rate": float(action_expert_learning_rate), "zeva_learning_rate": 5e-5,
        "prior_loss_weight": 0.01, "prior_residual_dropout_probability": 0.4,
        "memory_dropout": 0.1, "prior_injection_horizon": 50,
        "action_output_horizon": 50, "executed_horizon": 15, "seed": int(seed),
        "same_seed_and_data_order": True, "fresh_optimizer": True,
        "zero_initialized_dual_residual_projectors": True,
        "prior_nll_detach_context": arm in ("nll_detached", "nll_detached_full", "gradient_route_full", "gradient_route_smoke", "h15_route_full", "h15_route_smoke"),
        "decouple_action_expert_gradient": arm in ("gradient_route_full", "gradient_route_smoke", "h15_route_full", "h15_route_smoke"),
        "zeva_h15_flow_objective": arm in ("h15_route_full", "h15_route_smoke"),
        "smoke_only_no_checkpoint": arm in ("gradient_route_smoke", "h15_route_smoke"),
    },
    "lineage": {
        "handoff_root": str(Path(handoff).resolve()), "runtime_root": str(Path(runtime).resolve()),
        "foundation_checkpoint": str(Path(foundation).resolve()), "foundation_model_sha256": foundation_sha,
        "base_checkpoint": str(Path(base_checkpoint).resolve()), "base_model_sha256": base_sha,
        "base_manifest_sha256": base_manifest_sha,
        "stage1_language_checkpoint": str(Path(stage1_language).resolve()),
        "dataset_root": str(Path(dataset_root).resolve()), "task_subset": str(Path(task_subset).resolve()),
        "zte_checkpoint": str(Path(zte).resolve()), "zte_sha256": zte_sha,
        "causal_bank": str(Path(bank).resolve()), "causal_bank_sha256": bank_sha,
        "live_queries": str(Path(live_queries).resolve()), "live_queries_sha256": live_queries_sha,
        "task_retrieval": str(Path(retrieval).resolve()), "task_retrieval_sha256": retrieval_sha,
        "dataset_adapter_sha256": adapter_sha,
        "initial_stage2_and_immutable_teacher": str(Path(base_checkpoint).resolve()),
        "old_zeva_adapter": "forbidden",
    },
    "runtime": {
        "python": sys.executable,
        "torch_version": __import__("torch").__version__,
        "torch_module_file": module_file("torch"),
        "transformers_module_file": module_file("transformers"),
        "mamba_ssm_module_file": module_file("mamba_ssm"),
        "torchcodec_module_file": module_file("torchcodec"),
        "native_transformers_runtime": str(Path(native_transformers).resolve()),
        "shared_dependencies": str(Path(shared_deps).resolve()),
        "compiled_dependencies": str(Path(compiled_deps).resolve()),
        "model_dependency_overlay": dependency_overlay or None,
        "model_ld_library_path": ld_library_path or None,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "hf_home": os.environ.get("HF_HOME", ""),
        "torch_home": os.environ.get("TORCH_HOME", ""),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR", ""),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR", ""),
        "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH", ""),
        "xdg_cache_home": os.environ.get("XDG_CACHE_HOME", ""),
        "system_runtime_symlink_mutation": False,
    },
    "validation": {
        "split": "validation5", "full_validation": True, "eval_batches": 1000000,
        "diagnostic_eval_batches": 0,
        "fixed_checkpoint_step": int(steps),
        "reports": ["H50 flow", "executed H15 flow", "current residual-off", "fixed trained Base"],
        "formal_success_labels_used": False,
        "larger_residual_norm_is_not_utility": True,
    },
}
Path(output).open("x", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

run_arm() {
  local arm=$1
  local output="$run_root/$arm"
  local compile_cache="$run_root/_compile-cache/$arm"
  local main_port
  local gate_probability
  local step_dir
  local -a branch_args
  main_port=$(port_for_arm "$arm")
  gate_probability=$(gate_probability_for_arm "$arm")

  check_idle_gpus
  check_port_free "$main_port"
  mkdir -p "$output" "$compile_cache/torchinductor" "$compile_cache/triton" "$compile_cache/cuda" "$compile_cache/xdg" "$compile_cache/torch"

  export TORCHINDUCTOR_CACHE_DIR="$compile_cache/torchinductor"
  export TRITON_CACHE_DIR="$compile_cache/triton"
  export CUDA_CACHE_PATH="$compile_cache/cuda"
  export TORCH_HOME="$compile_cache/torch"
  export XDG_CACHE_HOME="$compile_cache/xdg"
  mkdir -p "$compile_cache/huggingface"
  export HF_HOME="$compile_cache/huggingface"
  export LIBRARY_PATH="${CUDA_STUB_LIBRARY:-/usr/local/cuda/lib64/stubs}${LIBRARY_PATH:+:$LIBRARY_PATH}"
  write_launcher_manifest "$arm" "$output" "$main_port" "$gate_probability"
  cp "$contract_config" "$output/experiment_config.json"

  printf '%s\n' \
    "starting gate mechanism arm=$arm $(date --iso-8601=seconds)" \
    "run_root=$run_root" \
    "dataset_root=$dataset_root" \
    "CUDA_VISIBLE_DEVICES=$gpu_list processes=$num_processes port=$main_port" \
    "steps=$steps warmup_steps=$warmup_steps save_freq=$save_freq eval_batches=$eval_batches" \
    "training_variant=zeva initial_residual_gate_probability=$gate_probability" \
    "initial_stage2_checkpoint=$base_checkpoint" \
    "immutable_anchor_stage2_checkpoint=$base_checkpoint" \
    "base_model_sha256=$actual_base_sha256" \
    "old_zeva_adapter=forbidden" \
    "native_transformers_runtime=$native_transformers" \
    "model_ld_library_path=${model_ld_library_path:-<unset>}" \
    "compile_cache=$compile_cache" \
    | tee "$output/STARTED"

  branch_args=(
    --training-variant zeva
    --handoff-root "$handoff"
    --foundation-checkpoint "$foundation"
    --goal-embedding-checkpoint "$stage1_language"
    --initial-stage2-checkpoint "$base_checkpoint"
    --anchor-stage2-checkpoint "$base_checkpoint"
    --dataset-root "$dataset_root"
    --task-subset "$task_subset"
    --zte-checkpoint "$zte_checkpoint"
    --causal-bank "$causal_bank"
    --live-queries "$live_queries"
    --task-retrieval "$task_retrieval"
    --save-dir "$output"
    --steps "$steps"
    --warmup-steps "$warmup_steps"
    --save-freq "$save_freq"
    --batch-size "$batch_size"
    --gradient-accumulation-steps "$gradient_accumulation_steps"
    --num-workers "$num_workers"
    --video-backend torchcodec
    --decoder-threads 1
    --action-expert-learning-rate "$action_expert_learning_rate"
    --learning-rate "$zeva_learning_rate"
    --prior-loss-weight "$prior_loss_weight"
    --preserve-loss-weight "$preserve_loss_weight"
    --paired-improvement-margin "$paired_improvement_margin"
    --baseline-preserve-interval "$baseline_preserve_interval"
    --gate-regularization-weight "$gate_regularization_weight"
    --phase-noise-std "$phase_noise_std"
    --memory-dropout "$memory_dropout"
    --prior-residual-dropout-probability "$prior_residual_dropout_probability"
    --prior-injection-horizon "$prior_injection_horizon"
    --initial-residual-gate-probability "$gate_probability"
    --eval-batches "$eval_batches"
    --compile-model
    --compile-mode "$compile_mode"
    --seed "$seed"
  )

  if [[ "$arm" == nll_detached || "$arm" == nll_detached_full || "$arm" == gradient_route_full || "$arm" == gradient_route_smoke || "$arm" == h15_route_full || "$arm" == h15_route_smoke ]]; then
    branch_args+=(--prior-nll-detach-context)
  fi
  if [[ "$arm" == gradient_route_full || "$arm" == gradient_route_smoke || "$arm" == h15_route_full || "$arm" == h15_route_smoke ]]; then
    branch_args+=(--decouple-action-expert-gradient)
  fi
  if [[ "$arm" == h15_route_full || "$arm" == h15_route_smoke ]]; then
    branch_args+=(--zeva-h15-flow-objective)
  fi
  if [[ "$arm" == gradient_route_smoke || "$arm" == h15_route_smoke ]]; then
    branch_args+=(--no-save-checkpoints)
  fi
  "$python_bin" -c 'import torchcodec' >/dev/null 2>&1 || die "TorchCodec is unavailable in the selected runtime"
  "$python_bin" -m accelerate.commands.launch \
    --multi_gpu \
    --main_process_port "$main_port" \
    --num_machines 1 \
    --num_processes "$num_processes" \
    --mixed_precision no \
    "$zeva_root/scripts/train_robotwin_stage2.py" \
    "${branch_args[@]}" \
    >> "$output/train.log" 2>&1

  if [[ "$arm" == gradient_route_smoke || "$arm" == h15_route_smoke ]]; then
    require_file "$output/manifest.json"
    printf '%s\n' "completed four-rank one-step gradient-route smoke $(date --iso-8601=seconds)" \
      | tee "$output/SMOKE_COMPLETE"
    return
  fi

  step_dir="$output/$(printf '%06d' "$steps")"
  require_file "$step_dir/model.safetensors"
  require_file "$step_dir/zeva_adapter.pth"
  require_file "$step_dir/training_state.pt"
  require_file "$output/manifest.json"
  [[ ! -e "$output/validation_diagnostics.json" ]] || die "refusing to overwrite diagnostics: $output/validation_diagnostics.json"

  # The post-training diagnostic is read-only and single-process.  It supplies
  # the H50/H15 residual-on/off/fixed-Base report that multi-GPU training
  # validation cannot emit with validation_diagnostics=true.
  check_idle_gpus
  CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "$python_bin" -u "$zeva_root/scripts/eval_robotwin_stage2_diagnostics.py" \
    --checkpoint "$step_dir" \
    --manifest "$output/manifest.json" \
    --output "$output/validation_diagnostics.json" \
    --fixed-teacher-checkpoint "$base_checkpoint" \
    --handoff-root "$handoff" \
    --dataset-root "$dataset_root" \
    --foundation-checkpoint "$foundation" \
    --goal-embedding-checkpoint "$stage1_language" \
    --zte-checkpoint "$zte_checkpoint" \
    --causal-bank "$causal_bank" \
    --live-queries "$live_queries" \
    --task-retrieval "$task_retrieval" \
    --batch-size "$batch_size" \
    --eval-batches 0 \
    --video-backend torchcodec \
    --decoder-threads 1 \
    --seed "$seed" \
    > "$output/validation_diagnostics.log" 2>&1

  "$python_bin" - "$output/validation_diagnostics.json" <<'PY'
import json
import math
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text())
protocol = report["protocol"]
expected = {"complete": True, "validation_decision_samples": 5874,
            "policy_horizon": 50, "executed_horizon": 15,
            "optimizer_created": False, "checkpoint_written": False}
for key, value in expected.items():
    if protocol.get(key) != value:
        raise SystemExit(f"Diagnostic protocol mismatch: {key}={protocol.get(key)!r}")
if not protocol.get("ordered_validation_samples_sha256"):
    raise SystemExit("Missing ordered validation sample identity")
if not math.isfinite(report["result"]["flow"]):
    raise SystemExit("Non-finite full validation flow")
if not report["fixed_teacher"].get("shared_frozen_weights_verified"):
    raise SystemExit("Fixed teacher frozen-weight verification missing")
PY

  printf '%s\n' "completed gate mechanism arm=$arm $(date --iso-8601=seconds)" \
    | tee "$output/COMPLETE"
}

for arm in "${arms[@]}"; do
  run_arm "$arm"
done

if [[ "$mode" == both ]]; then
  "$python_bin" - "$run_root" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
reports = [json.loads((root / arm / "validation_diagnostics.json").read_text())
           for arm in ("gate001", "gate010")]
keys = ("ordered_validation_samples_sha256", "seed", "batch_size", "dataset_adapter_sha256")
for key in keys:
    if reports[0]["protocol"].get(key) != reports[1]["protocol"].get(key):
        raise SystemExit(f"Gate arms have mismatched validation {key}")
PY
  printf '%s\n' "completed gate mechanism pair $(date --iso-8601=seconds)" \
    | tee "$run_root/COMPLETE"
fi
