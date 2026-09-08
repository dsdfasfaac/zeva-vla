#!/usr/bin/env bash
set -euo pipefail

# Calibrate the already-trained safe-router adapter on a disjoint closed-loop
# split, then run exactly one preregistered formal comparison.  Formal-test
# outcomes are never read by the scale selector.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model_host=${MODEL_HOST:-aigc29}
render_host=${RENDER_HOST:-aigc24}
model_ip=${MODEL_IP:-172.16.80.163}
source_checkpoint=${SOURCE_CHECKPOINT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-safe-router-v3/zeva/001250}
source_cache=${SOURCE_CACHE:-/data2/dingxin/zeva-checkpoint-cache-safe-router-v3/001250}
historical_root=${HISTORICAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5/formal-b1000-z250-seeded-v1}
failed_formal_root=${FAILED_FORMAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v3/formal-selected-validation5-v3}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-safe-router-v4-closed-loop}
cache_root=${CACHE_ROOT:-/data2/dingxin/zeva-checkpoint-cache-safe-router-v4}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
validation_episodes=${VALIDATION_EPISODES:-8}
validation_start_seed=${VALIDATION_START_SEED:-2000}
minimum_success_gain=${MINIMUM_SUCCESS_GAIN:-2}
scales=(0.25 0.5 1.0)

test -s "$source_checkpoint/zeva_adapter.pth"
test -s "$failed_formal_root/seed_manifest.json"
mkdir -p "$eval_root/configs" "$eval_root/calibrated-checkpoint"

python3 - "$failed_formal_root/state.json" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
if state.get("state") != "complete":
    raise RuntimeError("The prior formal run must finish before calibration starts.")
PY

make_candidate() {
  local scale=$1
  local tag=${scale/./}
  local cache=$cache_root/scale-$tag
  ssh "$model_host" "set -e; mkdir -p '$cache'; \
    ln -sfn '$source_cache/model.safetensors' '$cache/model.safetensors'; \
    python3 '$zeva_root/scripts/make_robotwin_residual_scale_candidate.py' \
      --adapter '$source_checkpoint/zeva_adapter.pth' \
      --output '$cache/zeva_adapter.pth' --scale '$scale'"
  python3 - "$eval_root/configs/scale-$tag.yml" "$cache" <<'PY'
import sys
from pathlib import Path

destination, checkpoint = sys.argv[1:]
Path(destination).write_text(f"""policy_name: zeva_policy
handoff_root: /mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
foundation_checkpoint: /data1/dingxin/zeva-checkpoint-cache/pretrained_model-best-v1
goal_embedding_checkpoint: /data1/dingxin/zeva-checkpoint-cache/pretrained_model-stage1-language-v1
stage2_checkpoint: {checkpoint}
zte_checkpoint: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/zte_best.pth
causal_bank: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/train_causal_bank.pt
retrieval_checkpoint: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1.5-task-retrieval/task_retrieval.pth
baseline_only: false
device: cuda
model_rng_seed: 20260907
""")
PY
}

for scale in "${scales[@]}"; do
  make_candidate "$scale"
done

baseline_config=$historical_root/../checkpoint-gates/b1000-z250-seeded-v1/configs/anchor.yml
first_scale=${scales[0]}
first_tag=${first_scale/./}
first_root=$eval_root/validation-scale-$first_tag

if [[ ! -f "$first_root/paired_report.json" ]]; then
  env MODEL_HOST="$model_host" RENDER_HOST="$render_host" MODEL_IP="$model_ip" \
    OUTPUT_ROOT="$first_root" BASELINE_CONFIG="$baseline_config" \
    ZEVA_CONFIG="$eval_root/configs/scale-$first_tag.yml" ANCHOR_CONFIG="" \
    TASK_MANIFEST="$task_manifest" EPISODES="$validation_episodes" \
    ABSOLUTE_START_SEED="$validation_start_seed" \
    BASELINE_LABEL="closed-loop-validation-pi05" \
    ZEVA_LABEL="closed-loop-validation-scale-$first_scale" \
    bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"
fi

baseline_successes=$(python3 -c "import json; print(json.load(open('$first_root/baseline/report.json'))['total_successes'])")
for scale in "${scales[@]:1}"; do
  tag=${scale/./}
  run_root=$eval_root/validation-scale-$tag
  [[ -f "$run_root/paired_report.json" ]] && continue
  env MODEL_HOST="$model_host" RENDER_HOST="$render_host" MODEL_IP="$model_ip" \
    OUTPUT_ROOT="$run_root" BASELINE_CONFIG="$baseline_config" \
    ZEVA_CONFIG="$eval_root/configs/scale-$tag.yml" ANCHOR_CONFIG="" \
    TASK_MANIFEST="$task_manifest" EPISODES="$validation_episodes" \
    ABSOLUTE_START_SEED="$validation_start_seed" \
    FROZEN_SEED_MANIFEST="$first_root/seed_manifest.json" \
    PRECOMPUTED_BASELINE_ROOT="$first_root/baseline" \
    PRECOMPUTED_BASELINE_EXPECTED_SUCCESSES="$baseline_successes" \
    BASELINE_LABEL="closed-loop-validation-pi05" \
    ZEVA_LABEL="closed-loop-validation-scale-$scale" \
    bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"
done

candidate_args=()
for scale in "${scales[@]}"; do
  tag=${scale/./}
  candidate_args+=(--candidate "$scale=$eval_root/validation-scale-$tag/zeva/progress")
done
python3 "$zeva_root/scripts/calibrate_robotwin_residual_trust.py" \
  --adapter "$source_checkpoint/zeva_adapter.pth" \
  --output "$eval_root/calibrated-checkpoint/zeva_adapter.pth" \
  --baseline-progress "$first_root/baseline/progress" \
  "${candidate_args[@]}" \
  --validation-seed-manifest "$first_root/seed_manifest.json" \
  --forbidden-seed-manifest "$failed_formal_root/seed_manifest.json" \
  --minimum-success-gain "$minimum_success_gain"

calibrated_cache=$cache_root/calibrated
ssh "$model_host" "set -e; mkdir -p '$calibrated_cache'; \
  ln -sfn '$source_cache/model.safetensors' '$calibrated_cache/model.safetensors'; \
  cp '$eval_root/calibrated-checkpoint/zeva_adapter.pth' '$calibrated_cache/zeva_adapter.pth'"
python3 - "$eval_root/configs/calibrated.yml" "$calibrated_cache" <<'PY'
import sys
from pathlib import Path

destination, checkpoint = sys.argv[1:]
Path(destination).write_text(f"""policy_name: zeva_policy
handoff_root: /mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
foundation_checkpoint: /data1/dingxin/zeva-checkpoint-cache/pretrained_model-best-v1
goal_embedding_checkpoint: /data1/dingxin/zeva-checkpoint-cache/pretrained_model-stage1-language-v1
stage2_checkpoint: {checkpoint}
zte_checkpoint: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/zte_best.pth
causal_bank: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1-zte/train_causal_bank.pt
retrieval_checkpoint: /mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1/stage1.5-task-retrieval/task_retrieval.pth
baseline_only: false
device: cuda
model_rng_seed: 20260907
""")
PY

formal_root=$eval_root/formal-calibrated-v4
env MODEL_HOST="$model_host" RENDER_HOST="$render_host" MODEL_IP="$model_ip" \
  OUTPUT_ROOT="$formal_root" BASELINE_CONFIG="$baseline_config" \
  ZEVA_CONFIG="$eval_root/configs/calibrated.yml" ANCHOR_CONFIG="" \
  BASELINE_IS_UNTOUCHED_ANCHOR=true \
  PRECOMPUTED_BASELINE_ROOT="$historical_root/anchor" \
  PRECOMPUTED_BASELINE_EXPECTED_SUCCESSES=114 \
  FROZEN_SEED_MANIFEST="$historical_root/seed_manifest.json" \
  TASK_MANIFEST="$task_manifest" \
  BASELINE_LABEL="untouched-pi05-best-v1-preregistered" \
  ZEVA_LABEL="safe-router-v4-closed-loop-calibrated" \
  MIN_BASELINE_SUCCESS_RATE=0.57 \
  bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

python3 "$zeva_root/scripts/robotwin_eval/audit_advantage10_formal.py" \
  "$formal_root" --require-accepted
