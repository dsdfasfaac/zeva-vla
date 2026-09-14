#!/usr/bin/env bash
# Fixed step-1000 evaluation; resource placement only, no checkpoint selection.
set -euo pipefail

zeva_release=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
zeva_runs=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang
zeva_staging=$zeva_runs/eval/formal-fixed-anchor-pair-20260914-staging
export OUTPUT_ROOT=$zeva_runs/eval/formal-fixed-anchor-pair-20260914
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Refusing existing evaluation output" >&2; exit 2; }
zeva_placement=${EVAL_PLACEMENT:-aigc24}
case "$zeva_placement" in
  aigc24)
    zeva_model_ip=172.16.80.158
    zeva_uuid2=GPU-4e851e19-19df-1b0c-cba8-4e0a81c08425
    zeva_uuid6=GPU-2e145d56-44e7-7643-311c-be4bdfb69eb0
    zeva_physical_gpus=(2 6)
    zeva_physical_uuids=("$zeva_uuid2" "$zeva_uuid6")
    zeva_slot_gpu_ids=2,6,2,6,2,6,2,6
    zeva_slot_cuda_devices=$zeva_uuid2,$zeva_uuid6,$zeva_uuid2,$zeva_uuid6,$zeva_uuid2,$zeva_uuid6,$zeva_uuid2,$zeva_uuid6
    zeva_memory_limit=1024
    zeva_health_proof=$zeva_release/renderer-formal-health-proof.json
    zeva_health_block=$zeva_release/gpu-health-blocked.json
    zeva_render_runtime=/data1/dingxin/robotwin-formal-eval/RoboTwin
    zeva_icd=/usr/share/vulkan/icd.d/nvidia_icd.json
    zeva_allowed_stale_pid=""
    ;;
  aigc31)
    zeva_model_ip=172.16.80.165
    zeva_uuid2=GPU-1e870d39-ddd0-127e-11a2-fb875ce44995
    zeva_uuid6=GPU-18077766-169d-96b9-2d53-55d11349ddd5
    zeva_physical_gpus=(0 1 2 3 4 5 6 7)
    zeva_physical_uuids=(
      GPU-da0bafe1-add8-e332-399b-c2c50db124fe
      GPU-81a4acdc-ff03-5734-fdfb-fe5a86e609f9
      "$zeva_uuid2"
      GPU-c37ba47c-c3ed-30b3-2b36-5892793bc97f
      GPU-cd1b3ba2-89ba-cfda-9135-2c812887b668
      GPU-1a2b2557-12b8-ec49-31d6-ae3b185d3319
      "$zeva_uuid6"
      GPU-267a6ac4-aa8e-3688-f0fc-d2cf14bece60
    )
    zeva_slot_gpu_ids=0,1,2,3,4,5,6,7
    zeva_slot_cuda_devices=$(IFS=,; printf '%s' "${zeva_physical_uuids[*]}")
    # Only the measured ~3.8 GiB residual allocation is allowed, not a live job.
    zeva_memory_limit=6144
    zeva_allowed_stale_pid=2369486
    zeva_health_proof=$zeva_release/renderer-formal-health-proof-a31.json
    zeva_health_block=$zeva_release/gpu-health-blocked-a31.json
    zeva_render_runtime=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin
    zeva_icd=/etc/vulkan/icd.d/nvidia_icd.json
    export MODEL_DEPENDENCY_OVERLAY=${MODEL_DEPENDENCY_OVERLAY:?Specify the verified a31 model dependency overlay}
    export MODEL_LD_LIBRARY_PATH=${MODEL_LD_LIBRARY_PATH:?Specify the verified isolated CUDA library paths}
    ;;
  aigc28)
    zeva_model_ip=172.16.80.162
    zeva_uuid2=GPU-ecaaa0cf-4454-f076-588a-c1667591a8d3
    zeva_uuid6=GPU-3ccf761f-a31a-809b-90cd-cfdf9394277a
    zeva_physical_gpus=(0 1 2 3 4 5 6 7)
    zeva_physical_uuids=(
      GPU-c4d9aceb-c274-fb29-e145-f6c2ef560bdc
      GPU-1b146bd7-326f-3563-b97e-24a26e13fa09
      "$zeva_uuid2"
      GPU-045ff755-16e3-8b76-cd03-4cf0ae1996b6
      GPU-ef75e39b-bdd0-7f30-b1ae-199650d79299
      GPU-2adf85a6-fab7-9216-fb00-14b253a4cd3e
      "$zeva_uuid6"
      GPU-323fc89e-78b5-e87e-91c3-f10569ac4e56
    )
    zeva_slot_gpu_ids=0,1,2,3,4,5,6,7
    zeva_slot_cuda_devices=$(IFS=,; printf '%s' "${zeva_physical_uuids[*]}")
    zeva_memory_limit=1024
    zeva_allowed_stale_pid=""
    zeva_health_proof=$zeva_release/renderer-formal-health-proof-a28.json
    zeva_health_block=$zeva_release/gpu-health-blocked-a28.json
    zeva_render_runtime=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin
    zeva_icd=/etc/vulkan/icd.d/nvidia_icd.json
    export MODEL_DEPENDENCY_OVERLAY=${MODEL_DEPENDENCY_OVERLAY:?Specify the verified model dependency overlay}
    export MODEL_LD_LIBRARY_PATH=${MODEL_LD_LIBRARY_PATH:?Specify the verified isolated CUDA library paths}
    ;;
  *) echo "Unsupported EVAL_PLACEMENT" >&2; exit 2;;
esac
[[ $(hostname -s) == "$zeva_placement" ]] || { echo "Run on the selected placement host" >&2; exit 2; }
[[ ! -e "$zeva_health_block" ]] || {
  echo "GPU compute/renderer health is blocked; fresh verified recovery required" >&2; exit 2;
}

# Check every physical target, including graphics processes. Never stop them.
for zeva_index in "${!zeva_physical_gpus[@]}"; do
  zeva_gpu=${zeva_physical_gpus[$zeva_index]}
  zeva_row=$(timeout -k 2 15 nvidia-smi -i "$zeva_gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) || {
    echo "GPU $zeva_gpu health query failed or timed out; launch cancelled" >&2; exit 2;
  }
  IFS=, read -r zeva_memory zeva_utilization <<< "$zeva_row"
  zeva_memory=${zeva_memory//[[:space:]]/}
  zeva_utilization=${zeva_utilization//[[:space:]]/}
  [[ "$zeva_memory" =~ ^[0-9]+$ && "$zeva_utilization" =~ ^[0-9]+$ ]]
  (( zeva_memory < zeva_memory_limit && zeva_utilization <= 5 )) || {
    echo "GPU $zeva_gpu is no longer idle; launch cancelled" >&2; exit 2;
  }
  zeva_current_uuid=$(timeout -k 2 15 nvidia-smi -i "$zeva_gpu" --query-gpu=uuid --format=csv,noheader)
  zeva_expected_uuid=${zeva_physical_uuids[$zeva_index]}
  [[ "$zeva_current_uuid" == "$zeva_expected_uuid" ]] || {
    echo "Physical GPU identity changed; fresh device verification required" >&2; exit 2;
  }
  zeva_active_jobs=$(timeout -k 2 15 nvidia-smi -i "$zeva_gpu" -q -x | python3 -c '
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
root = ET.parse(sys.stdin)
for process in root.findall(".//processes/process_info"):
    name = process.findtext("process_name", "")
    pid = process.findtext("pid", "unknown")
    if sys.argv[1] and pid == sys.argv[1] and not Path("/proc", pid).exists():
        continue
    if name.rsplit("/", 1)[-1] != "Xorg":
        print(pid, name)
' "$zeva_allowed_stale_pid")
  [[ -z "$zeva_active_jobs" ]] || {
    echo "GPU $zeva_gpu has non-Xorg processes; launch cancelled: $zeva_active_jobs" >&2; exit 2;
  }
done

export ZEVA_ROOT=$zeva_release
export MODEL_HOST=$zeva_placement RENDER_HOST=$zeva_placement MODEL_IP=$zeva_model_ip
export SLOTS=8 MODEL_GPU_IDS=$zeva_slot_gpu_ids RENDER_GPU_IDS=$zeva_slot_gpu_ids
export MODEL_CUDA_DEVICES=$zeva_slot_cuda_devices
export RENDER_CUDA_DEVICES=$MODEL_CUDA_DEVICES
export BASE_PORT=19300 EPISODES=20 ABSOLUTE_START_SEED=1000
export MODEL_SEED_POLICY=continuous MODEL_RNG_SEED=20260907
export MIN_BASELINE_SUCCESS_RATE=0.57 BASELINE_IS_UNTOUCHED_ANCHOR=false
export REQUIRE_EXPLICIT_FOUNDATION=true
export FOUNDATION_MODEL_SHA256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
export READ_ONLY_RUNTIME=true
export NATIVE_TRANSFORMERS_RUNTIME=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
export SHARED_RUNTIME=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin
export RENDER_RUNTIME=$zeva_render_runtime
export RENDER_VULKAN_ICD=$zeva_icd
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

python3 - "$FROZEN_SEED_MANIFEST" "$TASK_MANIFEST" "$zeva_staging" "$zeva_health_proof" "$RENDER_SAPIEN_DEVICE" "$zeva_uuid2" "$zeva_uuid6" "$zeva_placement" "$zeva_icd" "${MODEL_DEPENDENCY_OVERLAY:-}" "${MODEL_LD_LIBRARY_PATH:-}" "$MODEL_CUDA_DEVICES" <<'PY'
import hashlib
import json
from pathlib import Path
import socket
import sys

seed, tasks, staging, smoke = map(Path, sys.argv[1:5])
proof = json.loads(smoke.read_text())
assert proof['passed'] and proof['physical_gpu_index'] == 6
assert proof['host'] == sys.argv[8]
assert proof['cuda_visible_devices'] == sys.argv[7]
assert proof['renderer_device'] == sys.argv[5]
assert proof['matmul_verified_uuids'] == list(dict.fromkeys(sys.argv[12].split(',')))
assert proof['renderer_exit_code'] == 0
assert proof['vk_icd_filenames'] == sys.argv[9]
if sys.argv[8] in ('aigc31', 'aigc28'):
    assert proof['model_dependency_overlay'] == sys.argv[10]
    assert proof['model_ld_library_path'] == sys.argv[11]
if sys.argv[8] == 'aigc28':
    # The serving functional probe writes its report only after both inferences.
    # Its SSH transport may close before returning an exit code; never invent 0.
    # Renderer clean exit is independently required above (DeviceLost guard).
    assert proof['model_inference_completed'] and proof['model_process_absent_afterwards']
    model_smoke = proof['model_smoke']
    assert model_smoke['passed']
    assert model_smoke['config_sha256'] == 'c35716ddd585be23f504170da76b0ceab4067ffb10af1b09ad307134ce330ca6'
    assert model_smoke['first_shape'] == model_smoke['second_shape'] == [50, 16]
    assert model_smoke['committed_horizon'] == 15 and model_smoke['recurrent_transitions'] == 1
    assert model_smoke['cuda_visible_devices'] in proof['matmul_verified_uuids']
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
