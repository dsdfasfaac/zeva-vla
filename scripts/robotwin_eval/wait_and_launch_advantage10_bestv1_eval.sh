#!/usr/bin/env bash
set -euo pipefail

# Wait for the paired ten-task training run, select checkpoints from the
# validation curve, then run an exact-seed three-way closed-loop audit:
# original PI0.5 anchor, ten-task baseline fine-tune, and ZeVA fine-tune.

zeva_root=${ZEVA_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA}
train_root=${TRAIN_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/advantage10-best-v1-pair-v2-native-tf5}
output=${OUTPUT_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
handoff=/mnt/100T/users/huangbingjia/egoscalecausalclip/handoffs/robotwin-memory-baseline-v1
foundation=/data1/dingxin/zeva-checkpoint-cache/pretrained_model-best-v1
goal_encoder=/data1/dingxin/zeva-checkpoint-cache/pretrained_model-stage1-language-v1
stage1_root=/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/stage1-artifacts-v1
zte=$stage1_root/stage1-zte/zte_best.pth
causal_bank=$stage1_root/stage1-zte/train_causal_bank.pt
retrieval=$stage1_root/stage1.5-task-retrieval/task_retrieval.pth
expected_steps=${EXPECTED_STEPS:-1000}
model_rng_seed=${MODEL_RNG_SEED:-20260907}

mkdir -p "$output/configs"
printf '{"state":"waiting_for_training","expected_steps":%d,"updated":"%s"}\n' \
  "$expected_steps" "$(date -Iseconds)" > "$output/state.json"

while true; do
  zeva_step=$(python3 - "$train_root/zeva/latest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(int(json.loads(path.read_text())["step"]) if path.is_file() else 0)
PY
)
  if (( zeva_step >= expected_steps )); then
    break
  fi
  launcher_pid=$(cat "$train_root/zeva_launcher.pid" 2>/dev/null || true)
  if [[ -n "$launcher_pid" ]] && ! kill -0 "$launcher_pid" 2>/dev/null; then
    printf '{"state":"training_failed","latest_step":%d,"expected_steps":%d,"updated":"%s"}\n' \
      "$zeva_step" "$expected_steps" "$(date -Iseconds)" > "$output/state.json"
    exit 1
  fi
  printf '{"state":"waiting_for_training","latest_step":%d,"expected_steps":%d,"updated":"%s"}\n' \
    "$zeva_step" "$expected_steps" "$(date -Iseconds)" > "$output/state.json"
  sleep 30
done

# latest.json is committed after the complete checkpoint directory.  Wait for
# the distributed launcher to exit as an additional guard against evaluating
# while the final rank is still tearing down.
launcher_pid=$(cat "$train_root/zeva_launcher.pid" 2>/dev/null || true)
while [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null; do
  sleep 10
done

printf '{"state":"selecting_checkpoints","updated":"%s"}\n' "$(date -Iseconds)" > "$output/state.json"
python3 - "$train_root" "$expected_steps" "$output/validation_curves.json" <<'PY'
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
expected = int(sys.argv[2])
destination = Path(sys.argv[3])
payload = {"schema": "zeva-advantage10-checkpoint-selection-v1", "expected_steps": expected}
for variant in ("baseline", "zeva"):
    rows = []
    for checkpoint in sorted((root / variant).glob("[0-9][0-9][0-9][0-9][0-9][0-9]")):
        state_path = checkpoint / "training_state.pt"
        if not state_path.is_file():
            continue
        state = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
        validation = {key: float(value) for key, value in state["validation"].items()}
        rows.append({"step": int(state["step"]), "checkpoint": str(checkpoint), "validation": validation})
        del state
        gc.collect()
    if not rows or rows[-1]["step"] < expected:
        raise RuntimeError(f"{variant} did not complete {expected} steps")
    eligible = [
        row for row in rows
        if math.isfinite(row["validation"]["flow"])
        and (
            variant == "baseline"
            or (
                row["validation"]["retrieval_accuracy"] >= 0.95
                and math.isfinite(row["validation"]["prior_std"])
                and row["validation"]["flow"] <= row["validation"]["baseline"] * 1.01
            )
        )
    ]
    if not eligible:
        raise RuntimeError(f"{variant} has no validation-safe checkpoint")
    selected = min(eligible, key=lambda row: (row["validation"]["flow"], -row["step"]))
    payload[variant] = {"curve": rows, "selected": selected}

zeva_rows = payload["zeva"]["curve"]
if len(zeva_rows) >= 2:
    previous, final = zeva_rows[-2:]
    payload["zeva"]["final_interval_relative_flow_change"] = (
        final["validation"]["flow"] - previous["validation"]["flow"]
    ) / previous["validation"]["flow"]
payload["selection_policy"] = (
    "minimum validation flow among finite checkpoints; ZeVA additionally requires "
    "retrieval_accuracy>=0.95 and injected flow<=same-checkpoint no-injection flow*1.01"
)
temporary = destination.with_suffix(destination.suffix + ".partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
os.replace(temporary, destination)
PY

baseline_checkpoint=$(python3 -c "import json; print(json.load(open('$output/validation_curves.json'))['baseline']['selected']['checkpoint'])")
zeva_checkpoint=$(python3 -c "import json; print(json.load(open('$output/validation_curves.json'))['zeva']['selected']['checkpoint'])")

python3 - "$output/configs" "$handoff" "$foundation" "$goal_encoder" \
  "$baseline_checkpoint" "$zeva_checkpoint" "$zte" "$causal_bank" "$retrieval" \
  "$model_rng_seed" <<'PY'
import sys
from pathlib import Path

import yaml

(config_root, handoff, foundation, goal_encoder, baseline_checkpoint,
 zeva_checkpoint, zte, causal_bank, retrieval, model_rng_seed) = sys.argv[1:]
model_rng_seed = int(model_rng_seed)
root = Path(config_root)
configs = {
    "anchor_model_config.yml": {
        "policy_name": "zeva_policy",
        "handoff_root": handoff,
        "foundation_checkpoint": foundation,
        "baseline_only": True,
        "device": "cuda",
        "model_rng_seed": model_rng_seed,
    },
    "baseline_model_config.yml": {
        "policy_name": "zeva_policy",
        "handoff_root": handoff,
        "foundation_checkpoint": foundation,
        "stage2_checkpoint": baseline_checkpoint,
        "baseline_only": True,
        "device": "cuda",
        "model_rng_seed": model_rng_seed,
    },
    "zeva_model_config.yml": {
        "policy_name": "zeva_policy",
        "handoff_root": handoff,
        "foundation_checkpoint": foundation,
        "goal_embedding_checkpoint": goal_encoder,
        "stage2_checkpoint": zeva_checkpoint,
        "zte_checkpoint": zte,
        "causal_bank": causal_bank,
        "retrieval_checkpoint": retrieval,
        "baseline_only": False,
        "device": "cuda",
        "model_rng_seed": model_rng_seed,
    },
}
for name, config in configs.items():
    (root / name).write_text(yaml.safe_dump(config, sort_keys=False))
PY

python3 - "$output/alignment_contract.json" "$task_manifest" "$baseline_checkpoint" \
  "$zeva_checkpoint" "$foundation" <<'PY'
import json
import sys
from pathlib import Path

destination, tasks, baseline, zeva, anchor = sys.argv[1:]
payload = {
    "schema": "zeva-robotwin-train-eval-alignment-v1",
    "tasks": tasks,
    "scene": "zeva_randomized",
    "instruction_type": "seen",
    "camera": "Large_D435 640x480; RGB CHW float32 [0,1]",
    "camera_order": ["head", "left_wrist", "right_wrist"],
    "state": "absolute Joint14",
    "model_action": "chunk-start-relative EEF16 [50,16]",
    "execution": "execute first 15 actions then replan",
    "gripper": "slots 14/15; 1=open",
    "seeds": "first 20 expert-valid seeds per task starting at 1000; exact replay across conditions",
    "episodes_per_task": 20,
    "videos": "enabled; exactly 20 per task and cross-checked against progress",
    "baseline_checkpoint": baseline,
    "zeva_checkpoint": zeva,
    "anchor_foundation": anchor,
}
Path(destination).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

printf '{"state":"running_checkpoint_gate","updated":"%s"}\n' "$(date -Iseconds)" > "$output/state.json"
if ! bash "$zeva_root/scripts/robotwin_eval/run_advantage10_checkpoint_gate.sh" \
  "$baseline_checkpoint" "$zeva_checkpoint" "$output/checkpoint-gate"; then
  printf '{"state":"checkpoint_gate_failed_try_another_candidate","updated":"%s"}\n' \
    "$(date -Iseconds)" > "$output/state.json"
  exit 2
fi

printf '{"state":"running_paired_closed_loop","updated":"%s"}\n' "$(date -Iseconds)" > "$output/state.json"
OUTPUT_ROOT="$output" \
TASK_MANIFEST="$task_manifest" \
BASELINE_CONFIG="$output/configs/baseline_model_config.yml" \
ANCHOR_CONFIG="$output/configs/anchor_model_config.yml" \
ZEVA_CONFIG="$output/configs/zeva_model_config.yml" \
BASELINE_LABEL="advantage10-baseline" \
ANCHOR_LABEL="pretrained-model-best-v1-anchor" \
ZEVA_LABEL="advantage10-zeva" \
bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

python3 - "$output" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "paired_report.json").read_text())
baseline = float(report["baseline_success_rate"])
zeva = float(report["zeva_success_rate"])
anchor = float(report["anchor_success_rate"])
accepted = baseline >= anchor and zeva > baseline
acceptance = {
    "schema": "zeva-advantage10-acceptance-v1",
    "accepted": accepted,
    "requirements": {
        "baseline_not_below_original_pi_anchor": baseline >= anchor,
        "zeva_strictly_above_trained_baseline": zeva > baseline,
    },
    "success_rates": {"anchor": anchor, "baseline": baseline, "zeva": zeva},
    "next_action": "deliver" if accepted else "diagnose_and_continue_training_before_delivery",
}
temporary = root / "acceptance.json.partial"
temporary.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "acceptance.json")
(root / "state.json").write_text(json.dumps({
    "state": "accepted" if accepted else "needs_optimization",
    "acceptance": str(root / "acceptance.json"),
}, indent=2) + "\n")
if not accepted:
    raise SystemExit(2)
PY
