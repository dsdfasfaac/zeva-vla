#!/usr/bin/env python3
"""Select a frozen-PI v10 adapter using train95/validation5 metrics only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from select_robotwin_anchored_v9 import (
    FOUNDATION_SHA256,
    TASK_COUNT,
    checkpoints,
    finite,
    load_training_metadata,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("foundation_checkpoint", type=Path)
    parser.add_argument("--minimum-retrieval-accuracy", type=float, default=0.95)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.5)
    args = parser.parse_args()

    candidate_root = args.candidate_root.resolve()
    foundation = args.foundation_checkpoint.resolve()
    foundation_model = foundation / "model.safetensors"
    if sha256(foundation_model) != FOUNDATION_SHA256:
        raise ValueError("v10 requires the untouched best-v1 foundation")

    rows: list[dict[str, Any]] = []
    for step, path in sorted(checkpoints(candidate_root).items()):
        state = load_training_metadata(path / "training_state.pt")
        manifest = state.get("manifest", {})
        validation = state.get("validation", {})
        errors: list[str] = []
        if state.get("schema") != "zeva-robotwin-stage2-frozen-foundation-training-state-v2":
            errors.append("state_schema")
        if int(state.get("step", -1)) != step:
            errors.append("step_mismatch")
        if manifest.get("training_variant") != "adapter":
            errors.append("training_variant")
        if manifest.get("effective_global_batch_size") != 256:
            errors.append("global_batch")
        identity = manifest.get("foundation_identity", {})
        if identity.get("model_sha256") != FOUNDATION_SHA256:
            errors.append("foundation_identity")
        if manifest.get("initial_stage2_identity") is not None:
            errors.append("initial_stage2_must_be_absent")
        if manifest.get("anchor_stage2_identity") is not None:
            errors.append("anchor_stage2_must_be_absent")
        scope = manifest.get("task_scope", {})
        if len(scope.get("task_names", ())) != TASK_COUNT:
            errors.append("task_count")
        preservation = manifest.get("paired_baseline_preservation", {})
        if preservation.get("teacher") != "current_student_residual_off":
            errors.append("paired_teacher")
        frozen = set(manifest.get("frozen", ()))
        required_frozen = {
            "pi05_paligemma_vision_tower",
            "pi05_paligemma_language_backbone",
            "pi05_gemma_action_expert",
            "pi05_action_input_output_and_time_projections",
            "zte",
            "causal_bank",
            "task_retrieval",
        }
        if not required_frozen.issubset(frozen):
            errors.append("frozen_contract")
        optimizer_groups = manifest.get("optimizer_groups", {})
        if set(optimizer_groups) != {"zeva"}:
            errors.append("optimizer_groups")
        per_task = validation.get("per_task_paired", {})
        if len(per_task) != TASK_COUNT or any(
            int(item.get("count", 0)) <= 0 for item in per_task.values()
        ):
            errors.append("incomplete_per_task_validation")
        requirements = {
            "retrieval": finite(validation.get("retrieval_accuracy"))
            and float(validation["retrieval_accuracy"]) >= args.minimum_retrieval_accuracy,
            "aggregate_improvement": finite(validation.get("paired_improvement"))
            and float(validation["paired_improvement"]) > 0,
            "win_fraction": finite(validation.get("paired_win_fraction"))
            and float(validation["paired_win_fraction"]) >= args.minimum_win_fraction,
            "every_task_nonregression": finite(validation.get("minimum_task_paired_improvement"))
            and float(validation["minimum_task_paired_improvement"]) >= 0,
            "base_teacher_flow_finite": finite(validation.get("baseline")),
            "zeva_flow_finite": finite(validation.get("flow")),
        }
        row = {
            "step": step,
            "foundation_checkpoint": str(foundation),
            "training_foundation_checkpoint": identity.get("path"),
            "zeva_checkpoint": str(path),
            "base_validation_flow": validation.get("baseline"),
            "zeva_validation_flow": validation.get("flow"),
            "paired_improvement": validation.get("paired_improvement"),
            "paired_win_fraction": validation.get("paired_win_fraction"),
            "minimum_task_paired_improvement": validation.get("minimum_task_paired_improvement"),
            "per_task_paired": per_task,
            "retrieval_accuracy": validation.get("retrieval_accuracy"),
            "requirements": requirements,
            "errors": sorted(set(errors)),
        }
        row["eligible"] = not row["errors"] and all(requirements.values())
        rows.append(row)

    eligible = [row for row in rows if row["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda row: (
                -float(row["minimum_task_paired_improvement"]),
                -float(row["paired_improvement"]),
                -float(row["paired_win_fraction"]),
                float(row["zeva_validation_flow"]),
                int(row["step"]),
            ),
        )
        if eligible
        else None
    )
    selected_payload = None
    if selected is not None:
        checkpoint = Path(selected["zeva_checkpoint"])
        model_sha256 = sha256(checkpoint / "model.safetensors")
        if model_sha256 != FOUNDATION_SHA256:
            raise ValueError("selected frozen-PI checkpoint differs from untouched best-v1")
        selected_payload = {
            **selected,
            "artifacts": {
                "base_model_sha256": FOUNDATION_SHA256,
                "zeva_model_sha256": model_sha256,
                "zeva_adapter_sha256": sha256(checkpoint / "zeva_adapter.pth"),
            },
        }

    report = {
        "schema": "zeva-robotwin-foundation-adapter-v10-validation-selection-v1",
        "selection_data": "train95_validation5_only",
        "closed_loop_metrics_used": False,
        "fixed_base_checkpoint": str(foundation),
        "base_selected_by": "released untouched RoboTwin best-v1",
        "selection_order": [
            "maximum_minimum_task_paired_improvement",
            "maximum_aggregate_paired_improvement",
            "maximum_paired_win_fraction",
            "minimum_zeva_validation_flow",
            "earlier_step",
        ],
        "eligible_count": len(eligible),
        "rows": rows,
        "selected": selected_payload,
    }
    destination = candidate_root.parent / f"{candidate_root.name}-deployment-selection.json"
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "eligible_count": len(eligible),
                "selected_step": (selected_payload or {}).get("step"),
                "output": str(destination),
            },
            indent=2,
        )
    )
    if selected_payload is None:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
