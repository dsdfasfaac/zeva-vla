#!/usr/bin/env bash
set -euo pipefail

zeva_root=/mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA
train_root=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-conservative-ae5e-8-v1
cache_root=/data1/dingxin/zeva-checkpoint-cache-ae5e-8-v1
eval_parent=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5
seed_source_root=$eval_parent/formal-b1000-z250-seeded-v1
output=$eval_parent/formal-conservative-ae5e8-b125-z250-seeded-v1

while [[ ! -f "$train_root/baseline/COMPLETE" || ! -f "$train_root/zeva/COMPLETE" ]]; do
  sleep 20
done

mkdir -p "$cache_root"
copy_checkpoint() {
  local source=$1
  local name=$2
  local destination=$cache_root/$name
  if [[ ! -d "$destination" ]]; then
    local partial=$cache_root/.${name}.partial
    mkdir -p "$partial"
    rsync -a "$source/" "$partial/"
    mv "$partial" "$destination"
  fi
}
copy_checkpoint "$train_root/baseline/000125" baseline-000125
copy_checkpoint "$train_root/zeva/000250" zeva-000250

python3 - "$train_root" "$cache_root" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

train_root, cache_root = map(Path, sys.argv[1:])
pairs = [
    (train_root / "baseline/000125", cache_root / "baseline-000125"),
    (train_root / "zeva/000250", cache_root / "zeva-000250"),
]
evidence = {}
for source, destination in pairs:
    expected = {p.name: p.stat().st_size for p in source.iterdir() if p.is_file()}
    actual = {p.name: p.stat().st_size for p in destination.iterdir() if p.is_file()}
    if actual != expected:
        raise RuntimeError(f"cache file inventory mismatch: {source} -> {destination}")
    required = {"model.safetensors", "training_state.pt"}
    if "zeva" in destination.name:
        required.add("zeva_adapter.pth")
    if not required.issubset(actual):
        raise RuntimeError(f"missing required cached files in {destination}")
    evidence[destination.name] = actual
temporary = cache_root / "COMPLETE.partial"
temporary.write_text(json.dumps({"schema": "zeva-checkpoint-cache-v1", "files": evidence},
                                indent=2, sort_keys=True) + "\n")
os.replace(temporary, cache_root / "COMPLETE")
PY

# aigc24 is occupied by the first diagnostic formal run.  Reuse it only after
# that run has released all eight render GPUs; never overlap two RoboTwin
# processes on one rendering device.
while ! python3 - "$seed_source_root/state.json" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
raise SystemExit(0 if p.is_file() and json.loads(p.read_text()).get("state") == "complete" else 1)
PY
do
  sleep 30
done

exec env \
  MODEL_HOST=aigc32 \
  RENDER_HOST=aigc24 \
  MODEL_IP=172.16.80.166 \
  RENDER_RUNTIME=/data1/dingxin/robotwin-formal-eval/RoboTwin \
  RENDER_VULKAN_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json \
  SLOTS=8 \
  BASE_PORT=19500 \
  EPISODES=20 \
  OUTPUT_ROOT="$output" \
  BASELINE_CONFIG="$zeva_root/configs/robotwin_eval_advantage10_ae5e8_base125.yml" \
  ANCHOR_CONFIG="$zeva_root/configs/robotwin_eval_advantage10_conservative_anchor.yml" \
  ZEVA_CONFIG="$zeva_root/configs/robotwin_eval_advantage10_ae5e8_zeva250.yml" \
  TASK_MANIFEST="$zeva_root/configs/robotwin_zeva_advantage10.json" \
  FROZEN_SEED_MANIFEST="$seed_source_root/seed_manifest.json" \
  BASELINE_LABEL=advantage10-ae5e8-baseline-step125 \
  ANCHOR_LABEL=pretrained-model-best-v1-anchor \
  ZEVA_LABEL=advantage10-ae5e8-zeva-step250 \
  MODEL_RNG_SEED=20260907 \
  MODEL_SEED_POLICY=continuous \
  bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"
