#!/usr/bin/env bash
# Fixed step-1000 evaluation; resource placement only, no checkpoint selection.
set -euo pipefail

zeva_release=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
zeva_runs=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang
zeva_staging=$zeva_runs/eval/formal-fixed-anchor-pair-20260914-staging
export OUTPUT_ROOT=$zeva_runs/eval/formal-fixed-anchor-pair-20260914
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing existing evaluation output" >&2; exit 2; }
[[ $(hostname -s) == aigc24 ]] || { echo "Run this verified placement on aigc24" >&2; exit 2; }

# Only GPUs 2 and 6 were verified free. Other cards have graphics workloads
# despite an empty compute-apps query. Never stop those processes.
for zeva_gpu in 2 6; do
  zeva_row=$(timeout -k 2 15 nvidia-smi -i "$zeva_gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) || {
    echo "GPU $zeva_gpu health query failed or timed out; launch cancelled" >&2; exit 2;
  }
  IFS=, read -r zeva_memory zeva_utilization <<< "$zeva_row"
  zeva_memory=${zeva_memory//[[:space:]]/}
  zeva_utilization=${zeva_utilization//[[:space:]]/}
  [[ "$zeva_memory" =~ ^[0-9]+$ && "$zeva_utilization" =~ ^[0-9]+$ ]]
  (( zeva_memory < 1024 && zeva_utilization <= 5 )) || {
    echo "GPU $zeva_gpu is no longer idle; launch cancelled" >&2; exit 2;
  }
done

export ZEVA_ROOT=$zeva_release
export MODEL_HOST=aigc24 RENDER_HOST=aigc24 MODEL_IP=172.16.80.158
export SLOTS=8 MODEL_GPU_IDS=2,6,2,6,2,6,2,6 RENDER_GPU_IDS=2,6,2,6,2,6,2,6
export BASE_PORT=19300 EPISODES=20 ABSOLUTE_START_SEED=1000
export MODEL_SEED_POLICY=continuous MODEL_RNG_SEED=20260907
export MIN_BASELINE_SUCCESS_RATE=0.57 BASELINE_IS_UNTOUCHED_ANCHOR=false
export REQUIRE_EXPLICIT_FOUNDATION=true
export FOUNDATION_MODEL_SHA256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
export READ_ONLY_RUNTIME=true
export NATIVE_TRANSFORMERS_RUNTIME=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
export SHARED_RUNTIME=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin
export RENDER_RUNTIME=/data1/dingxin/robotwin-formal-eval/RoboTwin
export RENDER_VULKAN_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
# Set only after the saved PID-to-physical-GPU smoke below verifies the value.
export RENDER_SAPIEN_DEVICE=${RENDER_SAPIEN_DEVICE:?Specify the physically verified SAPIEN device selector}
export RENDER_WARP_CACHE_ROOT=$OUTPUT_ROOT/warp-cache
export FROZEN_SEED_MANIFEST=$zeva_runs/eval/formal-ztev2-selected-pair-20260912/seed_manifest.json
export TASK_MANIFEST=$zeva_release/configs/robotwin_zeva_advantage10.json
export BASELINE_CONFIG=$zeva_staging/robotwin_eval_ztev2_baseline.yml
export ZEVA_CONFIG=$zeva_staging/robotwin_eval_ztev2_zeva.yml
export ANCHOR_CONFIG=$zeva_staging/robotwin_eval_ztev2_anchor.yml
export BASELINE_LABEL=fixed-anchor-base-001000-h15
export ZEVA_LABEL=fixed-anchor-zeva-001000-h15
export ANCHOR_LABEL=untouched-best-v1-h15
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

python3 - "$FROZEN_SEED_MANIFEST" "$TASK_MANIFEST" "$zeva_staging" "$zeva_release/renderer-device-smoke.json" "$RENDER_SAPIEN_DEVICE" <<'PY'
import hashlib
import json
from pathlib import Path
import socket
import sys

seed, tasks, staging, smoke = map(Path, sys.argv[1:5])
proof = json.loads(smoke.read_text())
assert proof['passed'] and proof['physical_gpu_index'] == 6
assert proof['cuda_visible_devices'] == '6'
assert proof['renderer_device'] == sys.argv[5]
assert hashlib.sha256(seed.read_bytes()).hexdigest() == '1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b'
assert hashlib.sha256(tasks.read_bytes()).hexdigest() == '0501e43192415f5e32de993866536e25a3c9a51c2622248402bc3b9cd8b157bd'
manifest = json.loads((staging / 'robotwin_eval_ztev2_staging_manifest.json').read_text())
assert manifest['inputs']['base_stage2']['model_safetensors_sha256'] == 'bcf1d4f5f3e77926b8b7f798bb983378ad096e3892eed1a66058c5d98ed9bc17'
assert manifest['inputs']['zeva_stage2']['model_safetensors_sha256'] == 'f5e4812a01e01da9936ee23a8f2c9e7d5e70037112f821e759a329d37bc4d11a'
for port in range(19300, 19308):
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', port))
print('Pinned models, frozen task/seed/instruction manifests and ports verified')
PY

exec bash "$zeva_release/scripts/robotwin_eval/launch_paired_formal_eval.sh"
