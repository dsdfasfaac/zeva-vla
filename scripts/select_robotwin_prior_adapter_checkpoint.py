#!/usr/bin/env python3
"""Select a frozen-PI prior-adapter checkpoint using validation5 only.

The training loop keeps ``best.json`` for the lowest composite validation
loss.  That is useful for optimization diagnostics but is not the deployment
selection rule: a candidate must first demonstrate matched-PI non-regression
for every selected task.  This script re-derives that gate from every saved
``training_state.pt`` and rejects scheduler-drifted or structurally incomplete
checkpoints before ranking eligible candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


EXPECTED_STATE_SCHEMA = "zeva-robotwin-stage2-prior-only-training-state-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row["validation"]
    # ``min`` chooses the preferred row.  Match the preregistered ordering:
    # largest win rate, largest worst-task improvement, smallest degradation,
    # largest aggregate improvement, smallest prior NLL, earliest checkpoint.
    return (
        -float(metrics["paired_win_fraction"]),
        -float(metrics["minimum_task_paired_improvement"]),
        float(metrics["paired_degradation"]),
        -float(metrics["paired_improvement"]),
        float(metrics["prior"]),
        float(row["step"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--minimum-retrieval-accuracy", type=float, default=0.95)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-aggregate-improvement", type=float, default=0.0)
    parser.add_argument("--minimum-task-improvement", type=float, default=0.0)
    args = parser.parse_args()

    # Import lazily so ``--help`` and static inspection do not require the H100
    # training environment.
    import torch

    root = args.run_root.resolve()
    output = (args.output or root / "deployment_selection.json").resolve()
    candidates: list[dict[str, Any]] = []
    reference_manifest: dict[str, Any] | None = None

    for checkpoint in sorted(root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]")):
        row: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "step": int(checkpoint.name),
            "valid": False,
            "eligible": False,
            "errors": [],
            "ineligible_reasons": [],
        }
        state_path = checkpoint / "training_state.pt"
        adapter_path = checkpoint / "zeva_adapter.pth"
        model_path = checkpoint / "model.safetensors"
        for path in (state_path, adapter_path, model_path):
            if not path.is_file() or path.stat().st_size == 0:
                row["errors"].append(f"missing_or_empty:{path.name}")
        if row["errors"]:
            candidates.append(row)
            continue

        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("schema") != EXPECTED_STATE_SCHEMA:
            row["errors"].append(f"unexpected_schema:{state.get('schema')!r}")
        if int(state.get("step", -1)) != row["step"]:
            row["errors"].append("directory_and_state_step_mismatch")
        scheduler_last_epoch = int(state.get("scheduler_state_dict", {}).get("last_epoch", -1))
        row["scheduler_last_epoch"] = scheduler_last_epoch
        if scheduler_last_epoch != row["step"]:
            row["errors"].append("scheduler_global_step_drift")

        manifest = state.get("manifest", {})
        scheduler_contract = manifest.get("lr_scheduler", {})
        if scheduler_contract.get("accelerate_step_scheduler_with_optimizer") is not False:
            row["errors"].append("missing_corrected_scheduler_contract")
        if manifest.get("training_variant") != "prior_adapter":
            row["errors"].append("not_prior_adapter")
        if manifest.get("training_mode") != "frozen_pi05_zeva_prior_only_fixed_gate_v11":
            row["errors"].append("unexpected_training_mode")
        if reference_manifest is None:
            reference_manifest = manifest
        elif manifest != reference_manifest:
            row["errors"].append("manifest_changed_across_checkpoints")

        validation = state.get("validation", {})
        required = (
            "flow",
            "baseline",
            "prior",
            "retrieval_accuracy",
            "paired_improvement",
            "paired_win_fraction",
            "paired_degradation",
            "minimum_task_paired_improvement",
        )
        if not all(_finite(validation.get(key)) for key in required):
            row["errors"].append("missing_or_nonfinite_validation_metric")
        task_names = manifest.get("task_scope", {}).get("task_names", [])
        per_task = validation.get("per_task_paired", {})
        if set(per_task) != set(task_names) or not task_names:
            row["errors"].append("incomplete_validation_task_coverage")
        elif any(int(per_task[name].get("count", 0)) <= 0 for name in task_names):
            row["errors"].append("empty_validation_task")

        if row["errors"]:
            candidates.append(row)
            continue
        row["valid"] = True
        row["validation"] = validation
        row["artifact_identity"] = {
            "adapter_sha256": _sha256(adapter_path),
            "adapter_bytes": adapter_path.stat().st_size,
            "model_bytes": model_path.stat().st_size,
            "training_state_sha256": _sha256(state_path),
            "foundation_model_sha256_from_manifest": manifest.get("foundation_identity", {}).get(
                "model_sha256"
            ),
        }
        gates = {
            "retrieval_accuracy": float(validation["retrieval_accuracy"])
            >= args.minimum_retrieval_accuracy,
            "aggregate_improvement": float(validation["paired_improvement"])
            > args.minimum_aggregate_improvement,
            "paired_win_fraction": float(validation["paired_win_fraction"])
            >= args.minimum_win_fraction,
            "every_task_nonregression": float(validation["minimum_task_paired_improvement"])
            >= args.minimum_task_improvement,
        }
        row["gates"] = gates
        row["ineligible_reasons"] = [name for name, passed in gates.items() if not passed]
        row["eligible"] = all(gates.values())
        candidates.append(row)

    eligible = [row for row in candidates if row["eligible"]]
    selected = min(eligible, key=_selection_key) if eligible else None
    if selected is not None:
        # The full foundation is ~9 GB. Hash it only for the selected
        # checkpoint instead of multiplying shared-storage traffic by every
        # saved validation point.
        selected["artifact_identity"]["model_sha256"] = _sha256(
            Path(selected["checkpoint"]) / "model.safetensors"
        )
    payload = {
        "schema": "zeva-robotwin-prior-adapter-validation-selection-v1",
        "run_root": str(root),
        "selection_data": "train95_validation5_only",
        "formal_or_closed_loop_test_metrics_used": False,
        "criteria": {
            "minimum_retrieval_accuracy": args.minimum_retrieval_accuracy,
            "minimum_win_fraction": args.minimum_win_fraction,
            "minimum_aggregate_improvement_strict": args.minimum_aggregate_improvement,
            "minimum_task_improvement": args.minimum_task_improvement,
            "ranking": [
                "maximize_paired_win_fraction",
                "maximize_minimum_task_paired_improvement",
                "minimize_paired_degradation",
                "maximize_aggregate_paired_improvement",
                "minimize_prior_nll",
                "prefer_earlier_step",
            ],
        },
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "selected": (
            {
                "step": selected["step"],
                "checkpoint": selected["checkpoint"],
                "validation": selected["validation"],
                "artifact_identity": selected["artifact_identity"],
            }
            if selected is not None
            else None
        ),
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "eligible_count": len(eligible),
                "selected_step": selected["step"] if selected is not None else None,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
