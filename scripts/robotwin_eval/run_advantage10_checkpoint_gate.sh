#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 BASELINE_CHECKPOINT ZEVA_CHECKPOINT [OUTPUT_ROOT]" >&2
  exit 2
fi

baseline_checkpoint=$1
zeva_checkpoint=$2
zeva_root=${ZEVA_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA}
output=${3:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-native-tf5-checkpoint-gate}
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
foundation=${ROBOTWIN_FOUNDATION:-/data1/dingxin/zeva-checkpoint-cache/pretrained_model-best-v1}
goal_encoder=${ROBOTWIN_STAGE1_LANGUAGE:-/data1/dingxin/zeva-checkpoint-cache/pretrained_model-stage1-language-v1}
stage1_root=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1
zte=$stage1_root/stage1-zte/zte_best.pth
causal_bank=$stage1_root/stage1-zte/train_causal_bank.pt
retrieval=$stage1_root/stage1.5-task-retrieval/task_retrieval.pth
task_manifest=$zeva_root/configs/robotwin_zeva_gate3.json
seed_manifest=$zeva_root/configs/robotwin_zeva_gate3_seeds.json
model_rng_seed=${MODEL_RNG_SEED:-20260907}

for checkpoint in "$baseline_checkpoint" "$zeva_checkpoint"; do
  test -s "$checkpoint/model.safetensors" || {
    echo "missing complete model checkpoint: $checkpoint/model.safetensors" >&2
    exit 1
  }
done
test -s "$zeva_checkpoint/zeva_adapter.pth" || {
  echo "missing ZeVA adapter: $zeva_checkpoint/zeva_adapter.pth" >&2
  exit 1
}

mkdir -p "$output/configs"
python3 - "$output/configs" "$handoff" "$foundation" "$goal_encoder" \
  "$baseline_checkpoint" "$zeva_checkpoint" "$zte" "$causal_bank" "$retrieval" \
  "$model_rng_seed" <<'PY'
import sys
from pathlib import Path
import yaml

(root, handoff, foundation, goal_encoder, baseline_checkpoint,
 zeva_checkpoint, zte, causal_bank, retrieval, model_rng_seed) = sys.argv[1:]
model_rng_seed = int(model_rng_seed)
root = Path(root)
configs = {
    "anchor.yml": {
        "policy_name": "zeva_policy", "handoff_root": handoff,
        "foundation_checkpoint": foundation, "baseline_only": True, "device": "cuda",
        "model_rng_seed": model_rng_seed,
    },
    "baseline.yml": {
        "policy_name": "zeva_policy", "handoff_root": handoff,
        "foundation_checkpoint": foundation, "stage2_checkpoint": baseline_checkpoint,
        "baseline_only": True, "device": "cuda", "model_rng_seed": model_rng_seed,
    },
    "zeva.yml": {
        "policy_name": "zeva_policy", "handoff_root": handoff,
        "foundation_checkpoint": foundation, "goal_embedding_checkpoint": goal_encoder,
        "stage2_checkpoint": zeva_checkpoint, "zte_checkpoint": zte,
        "causal_bank": causal_bank, "retrieval_checkpoint": retrieval,
        "baseline_only": False, "device": "cuda", "model_rng_seed": model_rng_seed,
    },
}
for name, payload in configs.items():
    (root / name).write_text(yaml.safe_dump(payload, sort_keys=False))
PY

OUTPUT_ROOT="$output" \
TASK_MANIFEST="$task_manifest" \
FROZEN_SEED_MANIFEST="$seed_manifest" \
EPISODES=5 SLOTS=3 BASE_PORT=${BASE_PORT:-19300} MODEL_SEED_POLICY=continuous \
BASELINE_CONFIG="$output/configs/baseline.yml" \
ANCHOR_CONFIG="$output/configs/anchor.yml" \
ZEVA_CONFIG="$output/configs/zeva.yml" \
BASELINE_LABEL="checkpoint-gate-baseline" \
ANCHOR_LABEL="checkpoint-gate-anchor" \
ZEVA_LABEL="checkpoint-gate-zeva" \
bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

python3 - "$output" "$baseline_checkpoint" "$zeva_checkpoint" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "paired_report.json").read_text())
payload = {
    "schema": "zeva-native-tf5-checkpoint-gate-v1",
    "baseline_checkpoint": sys.argv[2],
    "zeva_checkpoint": sys.argv[3],
    "episodes": 15,
    "anchor_success_rate": report["anchor_success_rate"],
    "baseline_success_rate": report["baseline_success_rate"],
    "zeva_success_rate": report["zeva_success_rate"],
    "passed": (
        report["baseline_success_rate"] >= report["anchor_success_rate"]
        and report["zeva_success_rate"] > report["baseline_success_rate"]
    ),
    "note": "Pilot gate only; final acceptance requires all 10 tasks x 20 paired episodes.",
}
temporary = root / "checkpoint_gate.json.partial"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "checkpoint_gate.json")
print(json.dumps(payload, indent=2, sort_keys=True))
if not payload["passed"]:
    raise SystemExit(2)
PY
