#!/usr/bin/env bash
set -euo pipefail

# Matched fixed-Anchor Stage 2 development pair.
#
# Both branches start from the same completed Base/004500 weights and create a
# fresh optimizer.  The ``zeva`` branch additionally keeps an immutable copy of
# that same Base action path as its preservation anchor.  This launcher is
# deliberately self-contained: the historical 8-GPU wrapper creates/mutates a
# Transformers runtime symlink, so this script reproduces the verified runtime
# path ordering and invokes Accelerate directly.

mode=${1:-both}
case "$mode" in
  baseline|zeva|both) ;;
  *)
    echo "usage: $0 [baseline|zeva|both]" >&2
    exit 2
    ;;
esac

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
contract_config="$zeva_root/configs/robotwin_ztev2_fixed_anchor_pair_20260914.json"

# All paths are explicit and can be overridden for a verified host copy.  In
# particular, dataset_root is a data directory (not its source/cache parent).
handoff=${ROBOTWIN_HANDOFF:-/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1}
runtime=${ROBOTWIN_RUNTIME:-$handoff/runtime}
python_bin=${PI05_PYTHON:-python3}
native_transformers=${NATIVE_TRANSFORMERS_RUNTIME:-/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912}
shared_deps=${ZEVA_RUNTIME_DEPS:-/data1/dingxin/zeva-runtime-deps}
compiled_deps=${ZEVA_COMPILED_DEPS:-/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1}

foundation=${ROBOTWIN_FOUNDATION:-$handoff/checkpoint/pretrained_model-best-v1}
foundation_sha256=${ROBOTWIN_FOUNDATION_SHA256:-7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe}
stage1_language=${ROBOTWIN_STAGE1_LANGUAGE:-$handoff/checkpoint/pretrained_model}
dataset_root=${ROBOTWIN_DATASET_ROOT:-/data1/huangbingjia/robotwin-lerobot-sidney-eef16-v1/data}
task_subset=${ROBOTWIN_TASK_SUBSET:-$zeva_root/configs/robotwin_zeva_advantage10.json}
base_checkpoint=${ROBOTWIN_TRAINED_BASE:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-ztev2-schedulerfix-pair-20260911/baseline/004500}
base_sha256=${ROBOTWIN_TRAINED_BASE_SHA256:-2f106633403e5f2146bf7cd4f56858cbfdb856e9b2966d1c724a79ac4948c84f}
stage1_root=${ROBOTWIN_STAGE1_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-artifacts-scheduler-repaired-20260911}
zte_checkpoint=${ROBOTWIN_ZTE_CHECKPOINT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-zte-v2-phase-vector-mse-4096-20260911h-scheduler-repair-20260911i/zte_v2_step_004096.pth}
causal_bank=${ROBOTWIN_CAUSAL_BANK:-$stage1_root/train_causal_bank.pt}
live_queries=${ROBOTWIN_LIVE_QUERIES:-$stage1_root/live_queries_h15.pt}
task_retrieval=${ROBOTWIN_TASK_RETRIEVAL:-$stage1_root/task_retrieval.pth}
run_root=${RUN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/fixed-anchor-pair-20260914}

# The experiment settings below are intentionally fixed.  A smoke test should
# use smoke_robotwin_zeva_v2.py rather than silently changing this experiment's
# budget or validation coverage.
steps=1000
warmup_steps=100
save_freq=250
batch_size=16
gradient_accumulation_steps=4
num_processes=4
num_workers=${NUM_WORKERS:-4}
eval_batches=1000000
action_expert_learning_rate=5e-6
zeva_learning_rate=5e-5
prior_loss_weight=0.01
prior_residual_dropout_probability=0.4
baseline_preserve_interval=4
preserve_loss_weight=1.0
paired_improvement_margin=0.0
gate_regularization_weight=0.001
initial_residual_gate_probability=0.01
phase_noise_std=0.02
memory_dropout=0.1
prior_injection_horizon=50
compile_mode=default
seed=1000

gpu_list=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
gpu_memory_limit_mib=${GPU_MEMORY_LIMIT_MIB:-1024}
gpu_utilization_limit_pct=${GPU_UTILIZATION_LIMIT_PCT:-5}

case "$mode" in
  baseline) variants=(baseline) ;;
  zeva) variants=(zeva) ;;
  both) variants=(baseline zeva) ;;
esac

die() {
  echo "fixed-anchor pair preflight: $*" >&2
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

[[ -f "$contract_config" ]] || die "missing launcher contract: $contract_config"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required for checkpoint identity checks"
"$python_bin" --version >/dev/null 2>&1 || die "PI05_PYTHON is not executable: $python_bin"

# Keep the native overlay first, then the verified shared/compiled dependencies,
# system site, and handoff/project sources.  No directory or runtime symlink is
# created here; output and compiler caches are created below under RUN_ROOT.
handoff_runtime=$runtime
require_dir "$handoff_runtime"
require_dir "$native_transformers"
require_dir "$shared_deps"
require_dir "$compiled_deps"
system_site=$(
  "$python_bin" -c 'import site; print(site.getsitepackages()[0])'
)
export EGOSCALE_LEROBOT_SOURCE="$handoff_runtime/lerobot-main-py311-v1/src"
export PYTHONPATH="$native_transformers:$shared_deps:$compiled_deps:$system_site:$handoff_runtime/h100-extra-deps:$handoff_runtime/lerobot-overlay-v2:$EGOSCALE_LEROBOT_SOURCE:$handoff_runtime/src:$zeva_root/src:$zeva_root:$handoff_runtime/lerobot-main-deps-py311-v1"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=${LEROBOT_VIDEO_DECODER_CACHE_SIZE:-32}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

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

actual_foundation_sha256=$(sha256sum "$foundation/model.safetensors" | awk '{print $1}')
[[ "$actual_foundation_sha256" == "$foundation_sha256" ]] || die \
  "unexpected best-v1 foundation SHA256: $actual_foundation_sha256 (expected $foundation_sha256)"
actual_base_sha256=$(sha256sum "$base_checkpoint/model.safetensors" | awk '{print $1}')
[[ "$actual_base_sha256" == "$base_sha256" ]] || die \
  "unexpected Base/004500 model SHA256: $actual_base_sha256 (expected $base_sha256)"

# This experiment is intentionally fresh.  An old ZeVA adapter or optimizer
# state must never be implicitly picked up from a previous run.
[[ -z "${RESUME_CHECKPOINT:-}" ]] || die "RESUME_CHECKPOINT is forbidden; this is a fresh matched experiment"
[[ -z "${ZEVA_ADAPTER_CHECKPOINT:-}" ]] || die "ZEVA_ADAPTER_CHECKPOINT is forbidden; the adapter must be fresh"

IFS=',' read -r -a gpu_ids <<< "$gpu_list"
[[ "${#gpu_ids[@]}" -eq "$num_processes" ]] || die \
  "CUDA_VISIBLE_DEVICES=$gpu_list must list exactly $num_processes GPUs"
[[ "$gpu_memory_limit_mib" =~ ^[0-9]+$ ]] || die "GPU_MEMORY_LIMIT_MIB must be an integer"
[[ "$gpu_utilization_limit_pct" =~ ^[0-9]+$ ]] || die "GPU_UTILIZATION_LIMIT_PCT must be an integer"
[[ "$gpu_memory_limit_mib" -ge 0 ]] || die "GPU_MEMORY_LIMIT_MIB must be non-negative"
[[ "$gpu_utilization_limit_pct" -ge 0 ]] || die "GPU_UTILIZATION_LIMIT_PCT must be non-negative"
for gpu_id in "${gpu_ids[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || die "GPU id must be numeric: $gpu_id"
done
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required for the idle-GPU preflight"
export CUDA_VISIBLE_DEVICES="$gpu_list"

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

port_for_variant() {
  case "$1" in
    baseline) printf '%s\n' "${BASELINE_MAIN_PROCESS_PORT:-29584}" ;;
    zeva) printf '%s\n' "${ZEVA_MAIN_PROCESS_PORT:-29585}" ;;
    *) die "unknown variant for port selection: $1" ;;
  esac
}

# Refuse every branch/cache path before making any output.  This also catches a
# stale partial run, instead of treating it as a resumable experiment.
for variant in "${variants[@]}"; do
  output="$run_root/$variant"
  compile_cache="$run_root/_compile-cache/$variant"
  [[ ! -e "$output" ]] || die "refusing to overwrite existing output: $output"
  [[ ! -e "$compile_cache" ]] || die "refusing to reuse existing compile cache: $compile_cache"
done

run_variant() {
  local variant=$1
  local output="$run_root/$variant"
  local compile_cache="$run_root/_compile-cache/$variant"
  local main_port
  local -a branch_args
  main_port=$(port_for_variant "$variant")

  check_idle_gpus
  check_port_free "$main_port"
  mkdir -p "$output" "$compile_cache/torchinductor" "$compile_cache/triton" "$compile_cache/cuda" "$compile_cache/xdg"

  # These cache locations are per branch and live under /mnt/100T.  In
  # particular, do not let torch.compile, Triton, HF, or temporary decoding
  # spill into the nearly-full /data1 home/cache locations.
  export TORCHINDUCTOR_CACHE_DIR="$compile_cache/torchinductor"
  export TRITON_CACHE_DIR="$compile_cache/triton"
  export CUDA_CACHE_PATH="$compile_cache/cuda"
  export TORCH_HOME="$compile_cache/torch"
  export XDG_CACHE_HOME="$compile_cache/xdg"
  # Keep Unix-domain sockets and multiprocessing temporaries on the system
  # /tmp.  The compiler/HF/driver caches above remain isolated on /mnt/100T.
  export LIBRARY_PATH="${CUDA_STUB_LIBRARY:-/usr/local/cuda/lib64/stubs}${LIBRARY_PATH:+:$LIBRARY_PATH}"

  printf '%s\n' \
    "starting fixed-anchor pair branch=$variant $(date --iso-8601=seconds)" \
    "run_root=$run_root" \
    "dataset_root=$dataset_root" \
    "CUDA_VISIBLE_DEVICES=$gpu_list processes=$num_processes port=$main_port" \
    "steps=$steps warmup_steps=$warmup_steps save_freq=$save_freq eval_batches=$eval_batches" \
    "initial_stage2_checkpoint=$base_checkpoint" \
    "base_model_sha256=$actual_base_sha256" \
    "old_zeva_adapter=forbidden" \
    "compile_cache=$compile_cache" \
    | tee "$output/STARTED"
  cp "$contract_config" "$output/experiment_config.json"

  branch_args=(
    --training-variant "$variant"
    --handoff-root "$handoff"
    --foundation-checkpoint "$foundation"
    --initial-stage2-checkpoint "$base_checkpoint"
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
    --initial-residual-gate-probability "$initial_residual_gate_probability"
    --eval-batches "$eval_batches"
    --compile-model
    --compile-mode "$compile_mode"
    --seed "$seed"
  )
  if [[ "$variant" == zeva ]]; then
    branch_args+=(
      --goal-embedding-checkpoint "$stage1_language"
      --anchor-stage2-checkpoint "$base_checkpoint"
    )
  fi

  # Directly invoke the same validated Accelerate entry point used by the
  # trainer, without the legacy wrapper's runtime symlink mutations.
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

  printf '%s\n' "completed fixed-anchor pair branch=$variant $(date --iso-8601=seconds)" \
    | tee "$output/COMPLETE"
}

for variant in "${variants[@]}"; do
  run_variant "$variant"
done

if [[ "$mode" == both ]]; then
  printf '%s\n' "completed fixed-anchor pair $(date --iso-8601=seconds)" \
    | tee "$run_root/COMPLETE"
fi
