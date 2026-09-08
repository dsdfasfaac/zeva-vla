"""Offline non-regression gate for selective-action LIBERO Stage 3."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import torch
import tyro

from openpi.zeva.libero_contract import sha256


@dataclasses.dataclass
class Args:
    stage3_dir: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage3-selective-action"
    output_path: str | None = None
    required_final_step: int = 2_000
    minimum_retrieval_accuracy: float = 0.95
    maximum_relative_flow_regression: float = 0.0


def main(args: Args):
    root = Path(args.stage3_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "zeva-libero-stage3-selective-action-manifest-v1":
        raise ValueError("Not a selective-action LIBERO Stage 3 directory.")
    records = []
    for state_path in sorted(root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]/training_state.pt")):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        step_dir = state_path.parent
        action_path = step_dir / "stage3_action.pth"
        action = torch.load(action_path, map_location="cpu", weights_only=False) if action_path.is_file() else {}
        validation = state.get("validation", {})
        flow = float(validation.get("flow", math.inf))
        baseline = float(validation.get("baseline", math.inf))
        records.append({
            "step": int(state["step"]),
            "validation": {key: float(value) for key, value in validation.items()},
            "validation_loss": float(state.get("validation_loss", validation.get("total", math.inf))),
            "relative_flow_regression": (flow - baseline) / max(abs(baseline), 1e-12),
            "gradient_invariants_passed": bool(state.get("gradient_invariants_passed")),
            "complete": action_path.is_file() and state_path.is_file(),
            "action_schema": action.get("schema"),
            "checkpoint": str(step_dir),
        })
    if not records:
        raise FileNotFoundError("No LIBERO Stage 3 checkpoints were found.")
    best = min(records, key=lambda record: record["validation_loss"])
    declared = json.loads((root / "best.json").read_text())
    metrics = [value for record in records for value in record["validation"].values()]
    trainable = tuple(manifest.get("trainable", ()))
    selected_layers = {name.split(".layers.", 1)[1].split(".", 1)[0]
                       for name in trainable if ".gemma_expert.model.layers." in name}
    checks = {
        "training_complete": records[-1]["step"] >= args.required_final_step,
        "all_metrics_finite": all(math.isfinite(value) for value in metrics),
        "all_checkpoints_complete": all(record["complete"] for record in records),
        "action_schema_aligned": all(
            record["action_schema"] == "zeva-libero-stage3-selective-action-v1"
            for record in records
        ),
        "gradient_invariants_verified": all(
            record["gradient_invariants_passed"] for record in records
        ),
        "only_final_two_action_layers": selected_layers == {"16", "17"}
        and all("paligemma" not in name or "gemma_expert" in name for name in trainable),
        "task_retrieval_gate": best["validation"].get("retrieval_accuracy", 0.0)
        >= args.minimum_retrieval_accuracy,
        "matched_stage2a_non_regression": best["relative_flow_regression"]
        <= args.maximum_relative_flow_regression,
        "best_json_matches_scan": int(declared["step"]) == best["step"],
        "selected_action_checkpoint_hashable": sha256(
            Path(best["checkpoint"]) / "stage3_action.pth"
        ) != "",
    }
    result = {
        "schema": "zeva-libero-stage3-selective-action-gate-v1",
        "stage3_dir": str(root),
        "best": best,
        "checks": checks,
        "offline_passed": all(checks.values()),
        "closed_loop_gate_pending": True,
        "ready_for_paired_closed_loop_evaluation": all(checks.values()),
        "note": "Final model selection still requires paired Stage2A/Stage3 LIBERO rollouts.",
    }
    output = Path(args.output_path) if args.output_path else root / "stage3_gate.json"
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
