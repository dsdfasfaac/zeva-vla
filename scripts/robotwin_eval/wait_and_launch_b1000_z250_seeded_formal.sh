#!/usr/bin/env bash
set -euo pipefail

# Promote the reproducible three-task checkpoint gate to the complete
# ten-task x 20-episode paired RoboTwin evaluation.

zeva_root=${ZEVA_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-vla/ICML26-BehaviorVLA}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-best-v1-paired-v2-native-tf5}
gate_root=${GATE_ROOT:-$eval_root/checkpoint-gates/b1000-z250-seeded-v1}
formal_root=${FORMAL_ROOT:-$eval_root/formal-b1000-z250-seeded-v1}
task_manifest=${TASK_MANIFEST:-$zeva_root/configs/robotwin_zeva_advantage10.json}
model_rng_seed=${MODEL_RNG_SEED:-20260907}

mkdir -p "$formal_root"
printf '{"state":"waiting_for_seeded_gate","gate":"%s","updated":"%s"}\n' \
  "$gate_root" "$(date -Iseconds)" > "$eval_root/state.json"

gate_pid=$(cat "$gate_root/gate.pid")
while kill -0 "$gate_pid" 2>/dev/null; do
  sleep 30
done

python3 - "$gate_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "checkpoint_gate.json"
if not path.is_file():
    raise SystemExit(f"seeded gate did not produce {path}")
payload = json.loads(path.read_text())
for condition in ("baseline", "anchor", "zeva"):
    report_path = root / condition / "report.json"
    if not report_path.is_file():
        raise SystemExit(f"seeded gate is incomplete: missing {report_path}")
    report = json.loads(report_path.read_text())
    if report.get("total_episodes") != 15 or report.get("model_seed_policy") != "continuous":
        raise SystemExit(f"seeded gate has an invalid {condition} report: {report}")
    videos = list((root / condition / "results").glob("**/*.mp4"))
    if len(videos) != 15:
        raise SystemExit(f"seeded gate {condition} has {len(videos)} videos, expected 15")
# This three-task smoke gate catches loading/alignment failures, but RoboTwin's
# GPU physics is not bit-deterministic: repeated 15-episode anchor runs varied
# by three successes even after fixing the model RNG.  Do not turn that noisy
# pilot into the statistical acceptance test; the complete 200 paired episodes
# below are authoritative.
print(json.dumps({"gate_complete": True, "pilot_passed": bool(payload.get("passed"))}))
PY

printf '{"state":"running_seeded_formal_paired_eval","output":"%s","updated":"%s"}\n' \
  "$formal_root" "$(date -Iseconds)" > "$eval_root/state.json"

OUTPUT_ROOT="$formal_root" \
TASK_MANIFEST="$task_manifest" \
SLOTS=8 EPISODES=20 BASE_PORT=19200 \
MODEL_SEED_POLICY=continuous MODEL_RNG_SEED="$model_rng_seed" \
BASELINE_CONFIG="$gate_root/configs/baseline.yml" \
ANCHOR_CONFIG="$gate_root/configs/anchor.yml" \
ZEVA_CONFIG="$gate_root/configs/zeva.yml" \
BASELINE_LABEL="advantage10-baseline-step1000" \
ANCHOR_LABEL="pretrained-model-best-v1-anchor" \
ZEVA_LABEL="advantage10-zeva-step250" \
bash "$zeva_root/scripts/robotwin_eval/launch_paired_formal_eval.sh"

python3 - "$formal_root" "$eval_root/state.json" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
state_path = Path(sys.argv[2])
report = json.loads((root / "paired_report.json").read_text())
anchor = float(report["anchor_success_rate"])
baseline = float(report["baseline_success_rate"])
zeva = float(report["zeva_success_rate"])
accepted = baseline >= anchor and zeva > baseline
payload = {
    "schema": "zeva-advantage10-seeded-acceptance-v1",
    "accepted": accepted,
    "requirements": {
        "baseline_not_below_original_pi_anchor": baseline >= anchor,
        "zeva_strictly_above_trained_baseline": zeva > baseline,
    },
    "success_rates": {"anchor": anchor, "baseline": baseline, "zeva": zeva},
    "model_rng_seed": 20260907,
    "model_seed_policy": "continuous",
    "next_action": "deliver" if accepted else "diagnose_and_continue_training_before_delivery",
}
temporary = root / "acceptance.json.partial"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, root / "acceptance.json")
state_path.write_text(json.dumps({
    "state": "accepted" if accepted else "needs_optimization",
    "formal_root": str(root),
    "acceptance": str(root / "acceptance.json"),
    "updated": __import__("datetime").datetime.now().astimezone().isoformat(),
}, indent=2) + "\n")
if not accepted:
    raise SystemExit(2)
PY
