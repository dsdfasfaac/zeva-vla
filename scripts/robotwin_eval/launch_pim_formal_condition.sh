#!/usr/bin/env bash
# User-authorized three-way formal multi-attempt evaluation on the original frozen 10x20 manifest.
set -euo pipefail

condition=${1:-}
[[ "$condition" == base || "$condition" == parent || "$condition" == pim ]] || {
  echo 'usage: launch_pim_formal_condition.sh base|parent|pim' >&2; exit 2;
}
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-pim-20260919
formal_seed=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-fixed-anchor-pair-20260914/seed_manifest.json
task_manifest=$zeva_root/configs/robotwin_zeva_advantage10.json
old_formal=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-h15-route-selected-000500-a29-20260917/manifest.json
loader_dir=/mnt/100T/users/dingxin/VLA/runtime/zeva-vulkan-loader-a28-20260916
formal_sha=1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b

case "$condition" in
  base)
    expected_host=aigc29; model_ip=172.16.80.163; base_port=19900; reset_scope=episode
    config=$zeva_root/scripts/robotwin_eval/base1000_dev.yml
    label=base1000-formal-multiattempt-user-authorized
    output=$run/eval/formal-multiattempt-base-original10x20-user-authorized-20260920
    physical_ids=(0 1 2 3)
    physical_uuids=(GPU-dc037006-cc5d-131d-9b69-cae335dcf41f GPU-7d64ba08-7827-c2af-2cea-808b090e33db GPU-42dabc5e-22d7-fbef-fe82-08483690cf70 GPU-3919fd5d-0726-b99c-2516-bfe855577149)
    ;;
  parent)
    expected_host=aigc29; model_ip=172.16.80.163; base_port=19920; reset_scope=episode
    config=$zeva_root/scripts/robotwin_eval/parent_step5000_multiattempt.yml
    label=cte-bit-eap-parent-step5000-formal-multiattempt-user-authorized
    output=$run/eval/formal-multiattempt-parent-original10x20-user-authorized-20260920
    physical_ids=(4 5 6 7)
    physical_uuids=(GPU-23cf0ed2-0f7a-f425-7457-781b444b2811 GPU-e65459ab-c5dc-5535-7e5a-5004a3a8bcae GPU-356e1bc5-2ea9-e524-1c2e-ef7844140f13 GPU-06290db9-90c3-2137-8446-909c60a37420)
    ;;
  pim)
    expected_host=aigc28; model_ip=172.16.80.162; base_port=19940; reset_scope=attempt
    config=$zeva_root/scripts/robotwin_eval/pim_step2000_multiattempt.yml
    label=cte-bit-pim-eap-step2000-formal-multiattempt-user-authorized
    output=$run/eval/formal-multiattempt-pim-original10x20-user-authorized-20260920
    physical_ids=(0 1 2 3 4 5 6 7)
    physical_uuids=(GPU-c4d9aceb-c274-fb29-e145-f6c2ef560bdc GPU-1b146bd7-326f-3563-b97e-24a26e13fa09 GPU-ecaaa0cf-4454-f076-588a-c1667591a8d3 GPU-045ff755-16e3-8b76-cd03-4cf0ae1996b6 GPU-ef75e39b-bdd0-7f30-b1ae-199650d79299 GPU-2adf85a6-fab7-9216-fb00-14b253a4cd3e GPU-3ccf761f-a31a-809b-90cd-cfdf9394277a GPU-323fc89e-78b5-e87e-91c3-f10569ac4e56)
    ;;
esac

[[ $(hostname -s) == "$expected_host" ]] || { echo "$condition formal launch is pinned to $expected_host" >&2; exit 2; }
for required in "$formal_seed" "$task_manifest" "$old_formal" "$config" \
  "$loader_dir/libvulkan.so.1" "$loader_dir/libEGL.so.1"; do
  [[ -s "$required" ]] || { echo "Missing frozen formal input: $required" >&2; exit 2; }
done
[[ $(sha256sum "$formal_seed" | awk '{print $1}') == "$formal_sha" ]] || { echo 'Formal manifest SHA mismatch' >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Fresh formal output already exists: $output" >&2; exit 2; }
for index in "${!physical_ids[@]}"; do
  gpu=${physical_ids[$index]}
  IFS=, read -r uuid memory utilization < <(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
  [[ "$uuid" == "${physical_uuids[$index]}" ]] || exit 2
  (( memory < 1024 && utilization <= 5 )) || { echo "Target GPU$gpu became busy" >&2; exit 2; }
done
for port in $(seq "$base_port" $((base_port + 7))); do
  ! ss -ltn | grep -q ":$port " || { echo "Port $port is occupied" >&2; exit 2; }
done

slot_ids=(); slot_uuids=()
if (( ${#physical_ids[@]} == 4 )); then
  for index in "${!physical_ids[@]}"; do
    slot_ids+=("${physical_ids[$index]}" "${physical_ids[$index]}")
    slot_uuids+=("${physical_uuids[$index]}" "${physical_uuids[$index]}")
  done
else
  slot_ids=("${physical_ids[@]}"); slot_uuids=("${physical_uuids[@]}")
fi

model_ld=$(python3 - "$old_formal" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["model_ld_library_path"])
PY
)
mkdir -p "$output"
python3 - "$output/formal_authorization.json" "$condition" "$formal_sha" <<'PY'
import json, os, pathlib, sys
out=pathlib.Path(sys.argv[1])
payload={
  "schema":"zeva-pim-formal-user-authorization-v1",
  "condition":sys.argv[2],
  "seed_manifest_sha256":sys.argv[3],
  "authorized_after_development_gate_failure":True,
  "development_gate_result":"FAIL: PIM-parent +2/80 and first-attempt -1/80",
  "selection_from_formal_labels":False,
  "checkpoint_reselection_after_formal":False,
  "protocol":"original frozen 10x20, seen instruction, H50 output/H15 execution, max4 attempts",
}
tmp=out.with_suffix('.partial'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,out)
PY

export ZEVA_ROOT=$zeva_root MODEL_HOST=$expected_host RENDER_HOST=$expected_host MODEL_IP=$model_ip
export SLOTS=8 MODEL_GPU_IDS=$(IFS=,; echo "${slot_ids[*]}") RENDER_GPU_IDS=$(IFS=,; echo "${slot_ids[*]}")
export MODEL_CUDA_DEVICES=$(IFS=,; echo "${slot_uuids[*]}") RENDER_CUDA_DEVICES=$(IFS=,; echo "${slot_uuids[*]}")
export BASE_PORT=$base_port EPISODES=20 ABSOLUTE_START_SEED=1000
export MAX_POLICY_ATTEMPTS=4 BASELINE_ATTEMPT_RESET_SCOPE=$reset_scope ZEVA_ATTEMPT_RESET_SCOPE=$reset_scope
export MODEL_SEED_POLICY=continuous MODEL_RNG_SEED=20260907
export MIN_BASELINE_SUCCESS_RATE=0 BASELINE_IS_UNTOUCHED_ANCHOR=false
export PRECOMPUTED_BASELINE_ROOT='' REUSE_EXISTING_BASELINE=false BASELINE_ONLY_PRECOMPUTE=true
export FROZEN_SEED_MANIFEST=$formal_seed ANCHOR_CONFIG='' OUTCOME_TRACE_ENABLED=false
export REQUIRE_EXPLICIT_FOUNDATION=true FOUNDATION_MODEL_SHA256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
export READ_ONLY_RUNTIME=true
export NATIVE_TRANSFORMERS_RUNTIME=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
export MODEL_DEPENDENCY_OVERLAY=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
export MODEL_LD_LIBRARY_PATH=$model_ld
export SHARED_RUNTIME=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin RENDER_RUNTIME=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin
export RENDER_VULKAN_ICD=/etc/vulkan/icd.d/nvidia_icd.json RENDER_SAPIEN_DEVICE=cuda:0
export RENDER_LD_LIBRARY_PATH=$loader_dir:/usr/lib/x86_64-linux-gnu RENDER_WARP_CACHE_ROOT=$output/warp-cache
export TASK_MANIFEST=$task_manifest BASELINE_CONFIG=$config ZEVA_CONFIG=$config
export BASELINE_LABEL=$label ZEVA_LABEL=$label OUTPUT_ROOT=$output OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"
