#!/usr/bin/env bash
# Predeclared disjoint 10-task x 8 paired development rollout for epoch40 exploration.
set -euo pipefail
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[[ $(hostname -s) == aigc29 ]] || { echo 'Development launch is pinned to aigc29' >&2; exit 2; }
run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-behavior-effect-20260918
output=$run/eval/development-epoch040-step5000-a29-20260919
formal_seed=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-fixed-anchor-pair-20260914/seed_manifest.json
task_manifest=$zeva_root/configs/robotwin_zeva_advantage10.json
old_formal=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-h15-route-selected-000500-a29-20260917/manifest.json
loader_dir=/mnt/100T/users/dingxin/VLA/runtime/zeva-vulkan-loader-a28-20260916
expected_uuids=(
  GPU-dc037006-cc5d-131d-9b69-cae335dcf41f
  GPU-7d64ba08-7827-c2af-2cea-808b090e33db
  GPU-42dabc5e-22d7-fbef-fe82-08483690cf70
  GPU-3919fd5d-0726-b99c-2516-bfe855577149
  GPU-23cf0ed2-0f7a-f425-7457-781b444b2811
  GPU-e65459ab-c5dc-5535-7e5a-5004a3a8bcae
  GPU-356e1bc5-2ea9-e524-1c2e-ef7844140f13
  GPU-06290db9-90c3-2137-8446-909c60a37420
)
for required in "$formal_seed" "$task_manifest" "$old_formal" \
  "$loader_dir/libvulkan.so.1" "$loader_dir/libEGL.so.1" \
  "$zeva_root/scripts/robotwin_eval/baseline_base1000_behavior_effect_dev.yml" \
  "$zeva_root/scripts/robotwin_eval/behavior_effect_epoch040_step5000_dev.yml"; do
  [[ -s "$required" ]] || { echo "Missing development input: $required" >&2; exit 2; }
done
[[ $(sha256sum "$formal_seed" | awk '{print $1}') == 1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b ]] || exit 2
for index in "${!expected_uuids[@]}"; do
  IFS=, read -r uuid memory utilization < <(nvidia-smi -i "$index" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
  [[ "$uuid" == "${expected_uuids[$index]}" && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || exit 2
  (( memory < 1024 && utilization <= 5 )) || { echo "GPU$index became busy" >&2; exit 2; }
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || {
  echo 'Other compute process appeared; development launch cancelled' >&2; exit 2;
}
for port in $(seq 19600 19607); do
  ! ss -ltn | grep -q ":$port " || { echo "Port $port is occupied" >&2; exit 2; }
done

model_ld=$(python3 - "$old_formal" <<'PY'
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
print(p["model_ld_library_path"])
PY
)
export ZEVA_ROOT=$zeva_root MODEL_HOST=aigc29 RENDER_HOST=aigc29 MODEL_IP=172.16.80.163
export SLOTS=8 MODEL_GPU_IDS=0,1,2,3,4,5,6,7 RENDER_GPU_IDS=0,1,2,3,4,5,6,7
export MODEL_CUDA_DEVICES=$(IFS=,; echo "${expected_uuids[*]}") RENDER_CUDA_DEVICES=$(IFS=,; echo "${expected_uuids[*]}")
export BASE_PORT=19600 EPISODES=8 ABSOLUTE_START_SEED=1000000
export MODEL_SEED_POLICY=continuous MODEL_RNG_SEED=20260907
export MIN_BASELINE_SUCCESS_RATE=0 BASELINE_IS_UNTOUCHED_ANCHOR=false
export PRECOMPUTED_BASELINE_ROOT='' ANCHOR_CONFIG='' OUTCOME_TRACE_ENABLED=false
export REQUIRE_EXPLICIT_FOUNDATION=true FOUNDATION_MODEL_SHA256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
export READ_ONLY_RUNTIME=true
export NATIVE_TRANSFORMERS_RUNTIME=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
export MODEL_DEPENDENCY_OVERLAY=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
export MODEL_LD_LIBRARY_PATH=$model_ld
export SHARED_RUNTIME=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin RENDER_RUNTIME=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin
export RENDER_VULKAN_ICD=/etc/vulkan/icd.d/nvidia_icd.json RENDER_SAPIEN_DEVICE=cuda:0
export RENDER_LD_LIBRARY_PATH=$loader_dir:/usr/lib/x86_64-linux-gnu RENDER_WARP_CACHE_ROOT=$output/warp-cache
export TASK_MANIFEST=$task_manifest
export BASELINE_CONFIG=$zeva_root/scripts/robotwin_eval/baseline_base1000_behavior_effect_dev.yml
export ZEVA_CONFIG=$zeva_root/scripts/robotwin_eval/behavior_effect_epoch040_step5000_dev.yml
export BASELINE_LABEL=base1000-development-h15 ZEVA_LABEL=behavior-effect-epoch040-step5000-development-h15
export OUTPUT_ROOT=$output OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

[[ ! -e "$output" ]] || { echo "Fresh development output exists: $output" >&2; exit 2; }
export BASELINE_ONLY_PRECOMPUTE=true REUSE_EXISTING_BASELINE=false FROZEN_SEED_MANIFEST=''
bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

python3 - "$output/seed_manifest.json" "$formal_seed" "$output/development_seed_audit.json" <<'PY'
import hashlib, json, os, pathlib, sys
dev_path, formal_path, out_path = map(pathlib.Path, sys.argv[1:])
dev, formal = json.loads(dev_path.read_text()), json.loads(formal_path.read_text())
if (dev.get("schema") != "robotwin-expert-valid-seeds-v1" or dev.get("start_seed") != 1000000
        or dev.get("episodes_per_task") != 8 or set(dev["tasks"]) != set(formal["tasks"])):
    raise SystemExit("Development manifest does not match preregistration")
overlap = {}
for task, rows in dev["tasks"].items():
    if len(rows) != 8 or len({int(x["seed"]) for x in rows}) != 8:
        raise SystemExit(f"Invalid development rows for {task}")
    shared = sorted({int(x["seed"]) for x in rows} & {int(x["seed"]) for x in formal["tasks"][task]})
    if shared: overlap[task] = shared
if overlap: raise SystemExit(f"Development/formal seed overlap: {overlap}")
payload={"schema":"zeva-behavior-effect-development-seed-audit-v1","status":"PASS",
         "rule":"first-eight-expert-valid-per-task-from-1000000","tasks":len(dev["tasks"]),
         "episodes":sum(map(len,dev["tasks"].values())),"overlap_with_formal":0,
         "development_manifest_sha256":hashlib.sha256(dev_path.read_bytes()).hexdigest(),
         "formal_manifest_sha256":hashlib.sha256(formal_path.read_bytes()).hexdigest()}
tmp=out_path.with_suffix('.partial'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,out_path)
PY

export BASELINE_ONLY_PRECOMPUTE=false REUSE_EXISTING_BASELINE=true
for index in "${!expected_uuids[@]}"; do
  IFS=, read -r uuid memory utilization < <(nvidia-smi -i "$index" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
  [[ "$uuid" == "${expected_uuids[$index]}" ]] || exit 2
  (( memory < 1024 && utilization <= 5 )) || { echo "GPU$index became busy before ZeVA pair" >&2; exit 2; }
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || {
  echo 'Another compute process appeared before ZeVA pair; stopping safely' >&2; exit 2;
}
bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

python3 - "$output/paired_report.json" "$output/development_gate.json" <<'PY'
import json, os, pathlib, sys
report=json.loads(pathlib.Path(sys.argv[1]).read_text())
base=sum(int(round(x["baseline_success_rate"]*8)) for x in report["tasks"])
zeva=sum(int(round(x["zeva_success_rate"]*8)) for x in report["tasks"])
payload={"schema":"zeva-behavior-effect-development-gate-v1","passed":zeva-base>=4,
         "baseline_successes":base,"zeva_successes":zeva,"extra_successes":zeva-base,
         "paired_episodes":report["total_paired_episodes"],"required_extra_successes":4,
         "next":"frozen_formal_once" if zeva-base>=4 else "do_not_run_formal"}
out=pathlib.Path(sys.argv[2]); tmp=out.with_suffix('.partial'); tmp.write_text(json.dumps(payload,indent=2)+'\n'); os.replace(tmp,out)
PY
