#!/usr/bin/env bash
# Freeze a new disjoint 10x8 single-attempt Base manifest before Parent/PIM outcomes.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[[ $(hostname -s) == aigc29 ]] || { echo 'Episode-PIM Base launch is pinned to aigc29' >&2; exit 2; }
run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-episode-pim-20260920
output=$run/eval/development-singleattempt-base-seed3000000
formal_seed=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-fixed-anchor-pair-20260914/seed_manifest.json
prior_dev=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-cte-bit-eap-20260918/eval/development-epoch040-step5000-a29-20260919/seed_manifest.json
prior_pim_dev=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-pim-20260919/eval/development-multiattempt-base-seed2000000/seed_manifest.json
task_manifest=$zeva_root/configs/robotwin_zeva_advantage10.json
old_formal=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-h15-route-selected-000500-a29-20260917/manifest.json
loader_dir=/mnt/100T/users/dingxin/VLA/runtime/zeva-vulkan-loader-a28-20260916
expected_uuids=(
  GPU-dc037006-cc5d-131d-9b69-cae335dcf41f GPU-7d64ba08-7827-c2af-2cea-808b090e33db
  GPU-42dabc5e-22d7-fbef-fe82-08483690cf70 GPU-3919fd5d-0726-b99c-2516-bfe855577149
  GPU-23cf0ed2-0f7a-f425-7457-781b444b2811 GPU-e65459ab-c5dc-5535-7e5a-5004a3a8bcae
  GPU-356e1bc5-2ea9-e524-1c2e-ef7844140f13 GPU-06290db9-90c3-2137-8446-909c60a37420)

for required in "$formal_seed" "$prior_dev" "$prior_pim_dev" "$task_manifest" "$old_formal" \
  "$loader_dir/libvulkan.so.1" "$loader_dir/libEGL.so.1" \
  "$zeva_root/scripts/robotwin_eval/base1000_dev.yml" \
  "$zeva_root/scripts/robotwin_eval/cte_bit_eap_parent_step5000_dev.yml"; do
  [[ -s $required ]] || { echo "Missing development input: $required" >&2; exit 2; }
done
[[ ! -e $output ]] || { echo "Fresh output already exists: $output" >&2; exit 2; }
for index in "${!expected_uuids[@]}"; do
  IFS=, read -r uuid memory utilization < <(nvidia-smi -i "$index" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
  [[ $uuid == "${expected_uuids[$index]}" ]] || exit 2
  (( memory < 1024 && utilization <= 5 )) || { echo "GPU$index became busy" >&2; exit 2; }
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || { echo 'Other compute process appeared' >&2; exit 2; }
for port in $(seq 20000 20007); do ! ss -ltn | grep -q ":$port " || { echo "Port $port occupied" >&2; exit 2; }; done

model_ld=$(python3 - "$old_formal" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["model_ld_library_path"])
PY
)
export ZEVA_ROOT=$zeva_root MODEL_HOST=aigc29 RENDER_HOST=aigc29 MODEL_IP=172.16.80.163
export SLOTS=8 MODEL_GPU_IDS=0,1,2,3,4,5,6,7 RENDER_GPU_IDS=0,1,2,3,4,5,6,7
export MODEL_CUDA_DEVICES=$(IFS=,; echo "${expected_uuids[*]}") RENDER_CUDA_DEVICES=$(IFS=,; echo "${expected_uuids[*]}")
export BASE_PORT=20000 EPISODES=8 ABSOLUTE_START_SEED=3000000 MAX_POLICY_ATTEMPTS=1
export BASELINE_ATTEMPT_RESET_SCOPE=episode ZEVA_ATTEMPT_RESET_SCOPE=episode
export MODEL_SEED_POLICY=continuous MODEL_RNG_SEED=20260907
export MIN_BASELINE_SUCCESS_RATE=0 BASELINE_IS_UNTOUCHED_ANCHOR=false
export PRECOMPUTED_BASELINE_ROOT='' ANCHOR_CONFIG='' OUTCOME_TRACE_ENABLED=false
export REQUIRE_EXPLICIT_FOUNDATION=true FOUNDATION_MODEL_SHA256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
export READ_ONLY_RUNTIME=true
export NATIVE_TRANSFORMERS_RUNTIME=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
export MODEL_DEPENDENCY_OVERLAY=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
export MODEL_LD_LIBRARY_PATH=$model_ld
export SHARED_RUNTIME=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin
export RENDER_RUNTIME=$SHARED_RUNTIME RENDER_VULKAN_ICD=/etc/vulkan/icd.d/nvidia_icd.json RENDER_SAPIEN_DEVICE=cuda:0
export RENDER_LD_LIBRARY_PATH=$loader_dir:/usr/lib/x86_64-linux-gnu RENDER_WARP_CACHE_ROOT=$output/warp-cache
export TASK_MANIFEST=$task_manifest
export BASELINE_CONFIG=$zeva_root/scripts/robotwin_eval/base1000_dev.yml
export ZEVA_CONFIG=$zeva_root/scripts/robotwin_eval/cte_bit_eap_parent_step5000_dev.yml
export BASELINE_LABEL=base1000-episode-pim-development ZEVA_LABEL=unused-parent
export OUTPUT_ROOT=$output OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export BASELINE_ONLY_PRECOMPUTE=true REUSE_EXISTING_BASELINE=false FROZEN_SEED_MANIFEST=''

bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

python3 - "$output/seed_manifest.json" "$formal_seed" "$prior_dev" "$prior_pim_dev" "$output/development_seed_audit.json" <<'PY'
import hashlib, json, os, pathlib, sys
dev_path, formal_path, prior_path, prior_pim_path, out_path = map(pathlib.Path, sys.argv[1:])
dev = json.loads(dev_path.read_text())
references = {
    "formal": json.loads(formal_path.read_text()),
    "prior_development": json.loads(prior_path.read_text()),
    "prior_pim_development": json.loads(prior_pim_path.read_text()),
}
if dev.get("schema") != "robotwin-expert-valid-seeds-v1" or dev.get("start_seed") != 3000000 or dev.get("episodes_per_task") != 8:
    raise SystemExit("Episode-PIM development manifest violates preregistration")
overlap = {}
for task, rows in dev["tasks"].items():
    seeds = {int(row["seed"]) for row in rows}
    if len(rows) != 8 or len(seeds) != 8:
        raise SystemExit(f"Invalid seed rows for {task}")
    for label, reference in references.items():
        shared = sorted(seeds & {int(row["seed"]) for row in reference["tasks"][task]})
        if shared: overlap[f"{label}:{task}"] = shared
if overlap: raise SystemExit(f"Development seed overlap: {overlap}")
payload = {
    "schema": "zeva-episode-pim-development-seed-audit-v1", "status": "PASS",
    "rule": "first-eight-expert-valid-per-task-from-3000000", "tasks": len(dev["tasks"]),
    "episodes": sum(map(len, dev["tasks"].values())),
    "overlap_with_formal": 0, "overlap_with_prior_development": 0,
    "overlap_with_prior_pim_development": 0,
    "development_manifest_sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
}
temporary = out_path.with_suffix(".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, out_path)
PY
