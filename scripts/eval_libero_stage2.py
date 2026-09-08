"""Artifact and matched-baseline advancement gate for LIBERO Stage 2A."""

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
    stage2_dir: str = "/data1/dingxin/zeva-runs/libero-v3-h5-lang/stage2a-adapter"
    output_path: str | None = None
    required_final_step: int = 5_000
    minimum_retrieval_accuracy: float = 0.95
    maximum_relative_flow_regression: float = 0.0


def main(args: Args):
    root = Path(args.stage2_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "zeva-libero-stage2a-manifest-v2":
        raise ValueError("Not a formal LIBERO Stage 2A directory.")
    records = []
    for state_path in sorted(root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]/training_state.pt")):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        step_dir = state_path.parent
        adapter_path = step_dir / "zeva_adapter.pth"
        adapter = torch.load(adapter_path, map_location="cpu", weights_only=False) if adapter_path.is_file() else {}
        validation = state.get("validation", {})
        flow = float(validation.get("flow", math.inf))
        baseline = float(validation.get("baseline", math.inf))
        regression = (flow - baseline) / max(abs(baseline), 1e-12)
        records.append({
            "step": int(state["step"]),
            "validation": {key: float(value) for key, value in validation.items()},
            "validation_loss": float(state.get("validation_loss", validation.get("total", math.inf))),
            "relative_flow_regression": regression,
            "gradient_invariants_passed": bool(state.get("gradient_invariants_passed")),
            "complete": adapter_path.is_file() and state_path.is_file(),
            "adapter_schema": adapter.get("schema"),
            "checkpoint": str(step_dir),
        })
    if not records:
        raise FileNotFoundError("No LIBERO Stage 2A checkpoints were found.")
    best = min(records, key=lambda value: value["validation_loss"])
    declared = json.loads((root / "best.json").read_text())
    all_values = [value for record in records for value in record["validation"].values()]
    checks = {
        "final_step_reached": records[-1]["step"] >= args.required_final_step,
        "all_metrics_finite": all(math.isfinite(value) for value in all_values),
        "all_checkpoints_complete": all(record["complete"] for record in records),
        "adapter_schema_aligned": all(
            record["adapter_schema"] == "zeva-libero-stage2-adapter-v2" for record in records
        ),
        "frozen_pi_zte_and_adapter_gradients_verified": all(
            record["gradient_invariants_passed"] for record in records
        ),
        "task_retrieval_gate": best["validation"].get("retrieval_accuracy", 0.0)
        >= args.minimum_retrieval_accuracy,
        "matched_frozen_pi_non_regression": best["relative_flow_regression"]
        <= args.maximum_relative_flow_regression,
        "best_json_matches_scan": int(declared["step"]) == best["step"],
        "selected_adapter_hashable": sha256(Path(best["checkpoint"]) / "zeva_adapter.pth") != "",
    }
    result = {
        "schema": "zeva-libero-stage2a-gate-v2",
        "stage2_dir": str(root),
        "best": best,
        "thresholds": {
            "minimum_retrieval_accuracy": args.minimum_retrieval_accuracy,
            "maximum_relative_flow_regression": args.maximum_relative_flow_regression,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(args.output_path) if args.output_path else root / "stage2_gate.json"
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
