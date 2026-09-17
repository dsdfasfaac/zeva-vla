#!/usr/bin/env bash
# Formal matched Base/ZeVA 10x20 rollout for the validation-selected H15 model.
# The wrapper never chooses a checkpoint, seed, task, or inference gate.
set -euo pipefail

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_root=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/h15-route-full-20260916-a29/h15_route_full
staging=$run_root/eval_staging_selected_000500
health=$run_root/renderer_health_a29_all8_20260916/health_proof.json
selection=$run_root/h15_candidate_selection.json
serving=$run_root/selected_model_serving_smoke.json
seed_manifest=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-fixed-anchor-pair-20260914/seed_manifest.json
task_manifest=$zeva_root/configs/robotwin_zeva_advantage10.json
old_formal=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-fixed-anchor-pair-20260914/manifest.json
loader_dir=/mnt/100T/users/dingxin/VLA/runtime/zeva-vulkan-loader-a28-20260916
output=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/formal-h15-route-selected-000500-a29-20260917
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

[[ $(hostname -s) == aigc29 ]] || { echo "Formal launch is pinned to aigc29" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "Refusing existing formal output: $output" >&2; exit 2; }
for required in "$health" "$selection" "$serving" "$seed_manifest" "$task_manifest" \
  "$old_formal" "$staging/robotwin_eval_ztev2_staging_manifest.json" \
  "$staging/robotwin_eval_ztev2_baseline.yml" "$staging/robotwin_eval_ztev2_zeva.yml" \
  "$loader_dir/libvulkan.so.1" "$loader_dir/libEGL.so.1"; do
  [[ -s "$required" ]] || { echo "Missing formal input: $required" >&2; exit 2; }
done
[[ $(sha256sum "$seed_manifest" | awk '{print $1}') == 1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b ]] || exit 2
[[ $(sha256sum "$task_manifest" | awk '{print $1}') == 0501e43192415f5e32de993866536e25a3c9a51c2622248402bc3b9cd8b157bd ]] || exit 2

python3 - "$health" "$selection" "$serving" "$staging" "$run_root" <<'PY'
import json
from pathlib import Path
import sys
health, selection, serving, staging, run_root = map(Path, sys.argv[1:])
h, s, v = (json.loads(path.read_text()) for path in (health, selection, serving))
if not (h.get("passed") is True and len(h.get("probes", [])) == 8
        and all(p.get("passed") is True and p.get("process_exit_code") == 0
                and p.get("pid_mapping_verified") is True for p in h["probes"])):
    raise SystemExit("All-eight SAPIEN renderer health proof is incomplete")
if (s.get("decision") != "selected" or s.get("selected_step") != 500
        or s.get("selected_checkpoint") != str(run_root / "000500")
        or s.get("test_success_labels_used") is not False):
    raise SystemExit("Validation-only candidate selection mismatch")
if (v.get("passed") is not True or v.get("stage2_checkpoint") != s["selected_checkpoint"]
        or v.get("recurrent_transitions") != 1
        or v.get("first_shape") != [50, 16] or v.get("second_shape") != [50, 16]):
    raise SystemExit("Selected serving contract has not passed")
stage = json.loads((staging / "robotwin_eval_ztev2_staging_manifest.json").read_text())
if (stage.get("status") != "staged_not_evaluated"
        or stage["inputs"]["zeva_stage2"]["path"] != s["selected_checkpoint"]):
    raise SystemExit("Formal configs do not target selected ZeVA")
base = json.loads((staging / "robotwin_eval_ztev2_baseline.yml").read_text())
zeva = json.loads((staging / "robotwin_eval_ztev2_zeva.yml").read_text())
if (base.get("stage2_checkpoint") != stage["inputs"]["base_stage2"]["path"]
        or zeva.get("stage2_checkpoint") != s["selected_checkpoint"]
        or base.get("baseline_only") is not True):
    raise SystemExit("Formal Base/ZeVA checkpoint configuration mismatch")
PY

for index in "${!expected_uuids[@]}"; do
  actual=$(nvidia-smi -i "$index" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
  IFS=, read -r uuid memory utilization <<< "$actual"
  uuid=${uuid//[[:space:]]/}
  memory=${memory//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  [[ "$uuid" == "${expected_uuids[$index]}" && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || exit 2
  (( memory < 1024 && utilization <= 5 )) || { echo "GPU$index is busy" >&2; exit 2; }
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || {
  echo "Other compute process appeared; formal launch cancelled" >&2; exit 2;
}
for port in $(seq 19400 19407); do
  ! ss -ltn | grep -q ":$port " || { echo "Port $port is occupied" >&2; exit 2; }
done

model_ld=$(python3 - "$old_formal" <<'PY'
import json
from pathlib import Path
import sys
payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["model_dependency_overlay"] == "/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914"
assert payload["model_host"] == "aigc28" and payload["render_host"] == "aigc28"
print(payload["model_ld_library_path"])
PY
)

export ZEVA_ROOT=$zeva_root
export MODEL_HOST=aigc29 RENDER_HOST=aigc29 MODEL_IP=172.16.80.163
export SLOTS=8 MODEL_GPU_IDS=0,1,2,3,4,5,6,7 RENDER_GPU_IDS=0,1,2,3,4,5,6,7
export MODEL_CUDA_DEVICES=$(IFS=,; echo "${expected_uuids[*]}")
export RENDER_CUDA_DEVICES=$MODEL_CUDA_DEVICES
export BASE_PORT=19400 EPISODES=20 ABSOLUTE_START_SEED=1000
export MODEL_SEED_POLICY=continuous MODEL_RNG_SEED=20260907
export MIN_BASELINE_SUCCESS_RATE=0 BASELINE_IS_UNTOUCHED_ANCHOR=false
export BASELINE_ONLY_PRECOMPUTE=false REUSE_EXISTING_BASELINE=false
export PRECOMPUTED_BASELINE_ROOT="" ANCHOR_CONFIG="" OUTCOME_TRACE_ENABLED=false
export REQUIRE_EXPLICIT_FOUNDATION=true
export FOUNDATION_MODEL_SHA256=7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe
export READ_ONLY_RUNTIME=true
export NATIVE_TRANSFORMERS_RUNTIME=/mnt/100T/users/dingxin/VLA/runtime/zeva-stage2-resume-aigc24-20260912
export MODEL_DEPENDENCY_OVERLAY=/mnt/100T/users/dingxin/VLA/runtime/zeva-model-runtime-a31-torch271-cu126-20260914
export MODEL_LD_LIBRARY_PATH=$model_ld
export SHARED_RUNTIME=/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin
export RENDER_RUNTIME=$SHARED_RUNTIME
export RENDER_VULKAN_ICD=/etc/vulkan/icd.d/nvidia_icd.json
export RENDER_SAPIEN_DEVICE=cuda:0
export RENDER_LD_LIBRARY_PATH=$loader_dir:/usr/lib/x86_64-linux-gnu
export RENDER_WARP_CACHE_ROOT=$output/warp-cache
export FROZEN_SEED_MANIFEST=$seed_manifest TASK_MANIFEST=$task_manifest
export BASELINE_CONFIG=$staging/robotwin_eval_ztev2_baseline.yml
export ZEVA_CONFIG=$staging/robotwin_eval_ztev2_zeva.yml
export BASELINE_LABEL=h15-route-base-001000
export ZEVA_LABEL=h15-route-zeva-000500
export OUTPUT_ROOT=$output
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

exec bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"
