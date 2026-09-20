#!/usr/bin/env bash
# Evaluate one frozen single-attempt development condition against the audited Base set.
set -euo pipefail

condition=${1:-}
[[ "$condition" == parent || "$condition" == episode-pim ]] || {
  echo 'usage: launch_episode_pim_development_condition.sh parent|episode-pim' >&2
  exit 2
}

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-episode-pim-20260920
base_root=$run/eval/development-singleattempt-base-seed3000000
task_manifest=$zeva_root/configs/robotwin_zeva_advantage10.json
old_formal=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-h15-route-selected-000500-a29-20260917/manifest.json
loader_dir=/mnt/100T/users/dingxin/VLA/runtime/zeva-vulkan-loader-a28-20260916
expected_seed_sha=3160b039b7c571850ef8187f61818c0aaeaa0a8ebc88c8c189b1cd8fcdc86103

if [[ "$condition" == parent ]]; then
  expected_host=aigc29
  model_ip=172.16.80.163
  base_port=20020
  config=$zeva_root/scripts/robotwin_eval/behavior_effect_epoch040_step5000_dev.yml
  label=cte-bit-eap-parent-step5000-singleattempt-development
  output=$run/eval/development-singleattempt-parent-seed3000000
  expected_uuids=(
    GPU-dc037006-cc5d-131d-9b69-cae335dcf41f GPU-7d64ba08-7827-c2af-2cea-808b090e33db
    GPU-42dabc5e-22d7-fbef-fe82-08483690cf70 GPU-3919fd5d-0726-b99c-2516-bfe855577149
    GPU-23cf0ed2-0f7a-f425-7457-781b444b2811 GPU-e65459ab-c5dc-5535-7e5a-5004a3a8bcae
    GPU-356e1bc5-2ea9-e524-1c2e-ef7844140f13 GPU-06290db9-90c3-2137-8446-909c60a37420
  )
else
  expected_host=aigc28
  model_ip=172.16.80.162
  base_port=20040
  config=$zeva_root/scripts/robotwin_eval/episode_pim_step2000_dev.yml
  label=cte-bit-episode-pim-eap-step2000-singleattempt-development
  output=$run/eval/development-singleattempt-episode-pim-seed3000000
  expected_uuids=(
    GPU-c4d9aceb-c274-fb29-e145-f6c2ef560bdc GPU-1b146bd7-326f-3563-b97e-24a26e13fa09
    GPU-ecaaa0cf-4454-f076-588a-c1667591a8d3 GPU-045ff755-16e3-8b76-cd03-4cf0ae1996b6
    GPU-ef75e39b-bdd0-7f30-b1ae-199650d79299 GPU-2adf85a6-fab7-9216-fb00-14b253a4cd3e
    GPU-3ccf761f-a31a-809b-90cd-cfdf9394277a GPU-323fc89e-78b5-e87e-91c3-f10569ac4e56
  )
fi

[[ $(hostname -s) == "$expected_host" ]] || {
  echo "$condition development launch is pinned to $expected_host" >&2
  exit 2
}
for required in "$base_root/baseline/report.json" "$base_root/baseline/state.json" \
  "$base_root/seed_manifest.json" "$base_root/development_seed_audit.json" \
  "$task_manifest" "$old_formal" "$config" \
  "$loader_dir/libvulkan.so.1" "$loader_dir/libEGL.so.1"; do
  [[ -s "$required" ]] || { echo "Missing frozen input: $required" >&2; exit 2; }
done
python3 - "$base_root/baseline/report.json" "$base_root/development_seed_audit.json" \
  "$base_root/seed_manifest.json" "$expected_seed_sha" <<'PY'
import hashlib
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
audit = json.loads(pathlib.Path(sys.argv[2]).read_text())
seed_path = pathlib.Path(sys.argv[3])
if report.get("total_episodes") != 80 or report.get("total_successes") != 40:
    raise SystemExit("Frozen Base result differs from audited 40/80")
if audit.get("status") != "PASS" or any(
    audit.get(key) != 0
    for key in (
        "overlap_with_formal",
        "overlap_with_prior_development",
        "overlap_with_prior_pim_development",
    )
):
    raise SystemExit("Seed disjointness audit is not PASS")
actual = hashlib.sha256(seed_path.read_bytes()).hexdigest()
if actual != sys.argv[4] or audit.get("development_manifest_sha256") != actual:
    raise SystemExit("Frozen single-attempt seed manifest SHA mismatch")
PY

[[ ! -e "$output" ]] || { echo "Fresh output already exists: $output" >&2; exit 2; }
for index in "${!expected_uuids[@]}"; do
  IFS=, read -r uuid memory utilization < <(
    nvidia-smi -i "$index" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits
  )
  uuid=${uuid//[[:space:]]/}
  memory=${memory//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  [[ "$uuid" == "${expected_uuids[$index]}" ]] || exit 2
  (( memory < 1024 && utilization <= 5 )) || { echo "GPU$index became busy" >&2; exit 2; }
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || {
  echo 'Other compute process appeared' >&2
  exit 2
}
for port in $(seq "$base_port" $((base_port + 7))); do
  ! ss -ltn | grep -q ":$port " || { echo "Port $port is occupied" >&2; exit 2; }
done

model_ld=$(python3 - "$old_formal" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["model_ld_library_path"])
PY
)
export ZEVA_ROOT=$zeva_root MODEL_HOST=$expected_host RENDER_HOST=$expected_host MODEL_IP=$model_ip
export SLOTS=8 MODEL_GPU_IDS=0,1,2,3,4,5,6,7 RENDER_GPU_IDS=0,1,2,3,4,5,6,7
export MODEL_CUDA_DEVICES=$(IFS=,; echo "${expected_uuids[*]}")
export RENDER_CUDA_DEVICES=$(IFS=,; echo "${expected_uuids[*]}")
export BASE_PORT=$base_port EPISODES=8 ABSOLUTE_START_SEED=3000000 MAX_POLICY_ATTEMPTS=1
export BASELINE_ATTEMPT_RESET_SCOPE=episode ZEVA_ATTEMPT_RESET_SCOPE=episode
export MODEL_SEED_POLICY=continuous MODEL_RNG_SEED=20260907
export MIN_BASELINE_SUCCESS_RATE=0 BASELINE_IS_UNTOUCHED_ANCHOR=true
export PRECOMPUTED_BASELINE_ROOT=$base_root/baseline
export PRECOMPUTED_BASELINE_EXPECTED_SUCCESSES=40
export FROZEN_SEED_MANIFEST=$base_root/seed_manifest.json
export REUSE_EXISTING_BASELINE=false BASELINE_ONLY_PRECOMPUTE=false ANCHOR_CONFIG='' OUTCOME_TRACE_ENABLED=false
export REQUIRE_EXPLICIT_FOUNDATION=true FOUNDATION_MODEL_SHA256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
export READ_ONLY_RUNTIME=true
export NATIVE_TRANSFORMERS_RUNTIME=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
export MODEL_DEPENDENCY_OVERLAY=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
export MODEL_LD_LIBRARY_PATH=$model_ld
export SHARED_RUNTIME=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin
export RENDER_RUNTIME=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin
export RENDER_VULKAN_ICD=/etc/vulkan/icd.d/nvidia_icd.json RENDER_SAPIEN_DEVICE=cuda:0
export RENDER_LD_LIBRARY_PATH=$loader_dir:/usr/lib/x86_64-linux-gnu RENDER_WARP_CACHE_ROOT=$output/warp-cache
export TASK_MANIFEST=$task_manifest
export BASELINE_CONFIG=$zeva_root/scripts/robotwin_eval/baseline_base1000_behavior_effect_dev.yml
export ZEVA_CONFIG=$config BASELINE_LABEL=base1000-singleattempt-development ZEVA_LABEL=$label
export OUTPUT_ROOT=$output OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"
