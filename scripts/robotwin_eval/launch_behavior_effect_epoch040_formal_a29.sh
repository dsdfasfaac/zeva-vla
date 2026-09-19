#!/usr/bin/env bash
# One-shot frozen 10-task x 20 formal pair after the preregistered development gate.
set -euo pipefail
zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[[ $(hostname -s) == aigc29 ]] || { echo 'Formal launch is pinned to aigc29' >&2; exit 2; }
run=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-behavior-effect-20260918
development=$run/eval/development-epoch040-step5000-a29-20260919
output=$run/eval/formal-behavior-effect-epoch040-step5000-a29-20260919
prior_formal=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-h15-route-selected-000500-a29-20260917
frozen_seed=$prior_formal/seed_manifest.json
precomputed_baseline=$prior_formal/baseline
task_manifest=$zeva_root/configs/robotwin_zeva_advantage10.json
old_manifest=$prior_formal/manifest.json
loader_dir=/mnt/100T/users/dingxin/VLA/runtime/zeva-vulkan-loader-a28-20260916
base_checkpoint=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/fixed-anchor-pair-20260914/baseline/001000
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
for required in "$development/development_gate.json" "$frozen_seed" \
  "$precomputed_baseline/report.json" "$old_manifest" "$task_manifest" \
  "$loader_dir/libvulkan.so.1" "$loader_dir/libEGL.so.1" \
  "$base_checkpoint/model.safetensors" \
  "$zeva_root/scripts/robotwin_eval/baseline_base1000_behavior_effect_dev.yml" \
  "$zeva_root/scripts/robotwin_eval/behavior_effect_epoch040_step5000_dev.yml"; do
  [[ -s "$required" ]] || { echo "Missing frozen formal input: $required" >&2; exit 2; }
done
[[ ! -e "$output" ]] || { echo "Formal output already exists: $output" >&2; exit 2; }
[[ $(sha256sum "$frozen_seed" | awk '{print $1}') == 1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b ]] || exit 2
[[ $(sha256sum "$base_checkpoint/model.safetensors" | awk '{print $1}') == bcf1d4f5f3e77926b8b7f798bb983378ad096e3892eed1a66058c5d98ed9bc17 ]] || exit 2
python3 - "$development/development_gate.json" "$precomputed_baseline/report.json" "$prior_formal" <<'PY'
import json, pathlib, sys
gate = json.loads(pathlib.Path(sys.argv[1]).read_text())
report = json.loads(pathlib.Path(sys.argv[2]).read_text())
prior = pathlib.Path(sys.argv[3])
if (gate.get("schema") != "zeva-behavior-effect-development-gate-v1"
        or gate.get("passed") is not True or gate.get("extra_successes", -999) < 4
        or gate.get("paired_episodes") != 80):
    raise SystemExit("The preregistered development gate did not pass")
if (report.get("condition") != "baseline" or report.get("total_episodes") != 200
        or report.get("total_successes") != 111
        or report.get("action_contract") != "eef16_h50_execute_h15"):
    raise SystemExit("The frozen precomputed Base1000 evidence is not the audited 111/200 anchor")
for row in report["tasks"]:
    if len(row["seeds"]) != 20:
        raise SystemExit(f"Incomplete baseline row: {row['task']}")
for task in (prior / "tasks.txt").read_text().splitlines():
    progress = json.loads((prior / "baseline" / "progress" / f"{task}.json").read_text())
    videos = list((prior / "baseline" / "results" / task).glob("episode*_randomized-true_success-*.mp4"))
    if not progress.get("complete") or len(progress["episode_results"]) != 20 or len(videos) != 20:
        raise SystemExit(f"Incomplete precomputed baseline evidence for {task}")
PY

for index in "${!expected_uuids[@]}"; do
  IFS=, read -r uuid memory utilization < <(nvidia-smi -i "$index" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
  [[ "$uuid" == "${expected_uuids[$index]}" && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || exit 2
  (( memory < 1024 && utilization <= 5 )) || { echo "GPU$index became busy" >&2; exit 2; }
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || {
  echo 'Other compute process appeared; formal launch cancelled' >&2; exit 2;
}
for port in $(seq 19700 19707); do
  ! ss -ltn | grep -q ":$port " || { echo "Port $port is occupied" >&2; exit 2; }
done

model_ld=$(python3 - "$old_manifest" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["model_ld_library_path"])
PY
)
export ZEVA_ROOT=$zeva_root MODEL_HOST=aigc29 RENDER_HOST=aigc29 MODEL_IP=172.16.80.163
export SLOTS=8 MODEL_GPU_IDS=0,1,2,3,4,5,6,7 RENDER_GPU_IDS=0,1,2,3,4,5,6,7
export MODEL_CUDA_DEVICES=$(IFS=,; echo "${expected_uuids[*]}") RENDER_CUDA_DEVICES=$(IFS=,; echo "${expected_uuids[*]}")
export BASE_PORT=19700 EPISODES=20 ABSOLUTE_START_SEED=1000
export MODEL_SEED_POLICY=continuous MODEL_RNG_SEED=20260907
export MIN_BASELINE_SUCCESS_RATE=0.555 BASELINE_IS_UNTOUCHED_ANCHOR=true
export PRECOMPUTED_BASELINE_ROOT=$precomputed_baseline PRECOMPUTED_BASELINE_EXPECTED_SUCCESSES=111
export FROZEN_SEED_MANIFEST=$frozen_seed
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
export BASELINE_LABEL=base1000-frozen-formal-h15 ZEVA_LABEL=behavior-effect-epoch040-step5000-frozen-formal-h15
export OUTPUT_ROOT=$output OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export OUTCOME_TRACE_ENABLED=false
bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

python3 - "$output" <<'PY'
import hashlib, json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
seed = root / "seed_manifest.json"
if hashlib.sha256(seed.read_bytes()).hexdigest() != "1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b":
    raise SystemExit("Formal manifest changed")
base = json.loads((root / "baseline" / "report.json").read_text())
zeva = json.loads((root / "zeva" / "report.json").read_text())
paired = json.loads((root / "paired_report.json").read_text())
base_successes, zeva_successes = base["total_successes"], zeva["total_successes"]
payload = {
    "schema": "zeva-behavior-effect-frozen-formal-gate-v1",
    "passed": zeva_successes - base_successes >= 8,
    "baseline_successes": base_successes,
    "zeva_successes": zeva_successes,
    "extra_successes": zeva_successes - base_successes,
    "paired_episodes": paired["total_paired_episodes"],
    "required_extra_successes": 8,
    "manifest_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
    "historical_formal_exposure_disclosed": True,
    "next": "deliver_with_full_audit" if zeva_successes - base_successes >= 8 else "goal_not_met",
}
if paired["total_paired_episodes"] != 200 or base_successes != 111:
    raise SystemExit("Formal evidence count changed")
for condition in ("baseline", "zeva"):
    videos = list((root / condition / "results").glob("*/episode*_randomized-true_success-*.mp4"))
    if len(videos) != 200:
        raise SystemExit(f"{condition}: expected 200 videos, got {len(videos)}")
out = root / "formal_gate.json"; tmp = out.with_suffix(".partial")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, out)
PY
