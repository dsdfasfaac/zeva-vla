#!/usr/bin/env bash
set -euo pipefail

# Run the two crossed seed x diffusion-RNG cells only after the first two cells
# pass, then audit all four cells before allowing a seed-1000 formal run.

zeva_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
step=${ZEVA_STEP:-500}
eval_root=${EVAL_ROOT:-/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/eval/advantage10-prior-action-expert-v11/step-$(printf '%06d' "$step")}
episodes=${EPISODES:-8}

python3 - "$eval_root/validation_summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
if summary.get("accepted_for_independent_confirmation") is not True:
    raise SystemExit("the first two v11 validation cells did not pass")
PY

EVAL_ROOT="$eval_root/cross" ZEVA_STEP="$step" EPISODES="$episodes" \
  SPLIT_I_SEED=15000 SPLIT_J_SEED=16000 \
  SPLIT_I_RNG=20260908 SPLIT_J_RNG=20260907 \
  bash "$zeva_root/scripts/robotwin_eval/launch_prior_action_expert_v11_validation.sh"

python3 "$zeva_root/scripts/robotwin_eval/audit_v11_four_cell_validation.py" \
  "$eval_root" --episodes-per-task "$episodes" --minimum-total-gain 12
