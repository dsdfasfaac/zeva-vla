#!/usr/bin/env python3
"""Select one shared validation-only step for the matched v8 Base/ZeVA pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


FOUNDATION_SHA256 = "7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe"
TASK_COUNT = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoints(root: Path) -> dict[int, Path]:
    result = {
        int(path.name): path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and (path / "training_state.pt").is_file()
        and (path / "model.safetensors").is_file()
    }
    if not result:
        raise RuntimeError(f"No complete checkpoints found under {root}")
    return result


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_manifest(base: dict[str, Any], zeva: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if base.get("training_variant") != "baseline":
        errors.append("base_variant")
    if zeva.get("training_variant") != "zeva":
        errors.append("zeva_variant")
    for name, manifest in (("base", base), ("zeva", zeva)):
        if manifest.get("effective_global_batch_size") != 256:
            errors.append(f"{name}_global_batch")
        if manifest.get("foundation_identity", {}).get("model_sha256") != FOUNDATION_SHA256:
            errors.append(f"{name}_foundation")
        frozen = set(manifest.get("frozen", ()))
        if not {
            "pi05_paligemma_vision_tower",
            "pi05_paligemma_language_backbone",
            "zte",
            "causal_bank",
            "task_retrieval",
        }.issubset(frozen):
            errors.append(f"{name}_frozen_contract")
        tasks = manifest.get("task_scope", {}).get("task_names", ())
        if len(tasks) != TASK_COUNT:
            errors.append(f"{name}_task_count")
        group = manifest.get("optimizer_groups", {}).get("pi05_action_expert", {})
        if float(group.get("learning_rate", -1)) != 5e-6:
            errors.append(f"{name}_action_expert_lr")
    if base.get("task_scope") != zeva.get("task_scope"):
        errors.append("task_scope_mismatch")
    if base.get("effective_global_batch_size") != zeva.get("effective_global_batch_size"):
        errors.append("batch_mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_root", type=Path)
    parser.add_argument("--minimum-retrieval-accuracy", type=float, default=0.95)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.5)
    args = parser.parse_args()

    root = args.pair_root.resolve()
    base_paths = checkpoints(root / "baseline")
    zeva_paths = checkpoints(root / "zeva")
    common_steps = sorted(set(base_paths).intersection(zeva_paths))
    if not common_steps:
        raise RuntimeError("Base and ZeVA have no complete shared checkpoint step.")

    rows: list[dict[str, Any]] = []
    for step in common_steps:
        base_state = torch.load(
            base_paths[step] / "training_state.pt", map_location="cpu", weights_only=False
        )
        zeva_state = torch.load(
            zeva_paths[step] / "training_state.pt", map_location="cpu", weights_only=False
        )
        errors: list[str] = []
        if base_state.get("schema") != "robotwin-pi05-action-expert-baseline-training-state-v1":
            errors.append("base_schema")
        if zeva_state.get("schema") != "zeva-robotwin-stage2-action-expert-training-state-v8":
            errors.append("zeva_schema")
        if int(base_state.get("step", -1)) != step or int(zeva_state.get("step", -1)) != step:
            errors.append("step_mismatch")
        errors.extend(validate_manifest(base_state.get("manifest", {}), zeva_state.get("manifest", {})))

        base_validation = base_state.get("validation", {})
        zeva_validation = zeva_state.get("validation", {})
        per_task = zeva_validation.get("per_task_paired", {})
        if len(per_task) != TASK_COUNT or any(int(row.get("count", 0)) <= 0 for row in per_task.values()):
            errors.append("incomplete_per_task_validation")
        requirements = {
            "retrieval": finite(zeva_validation.get("retrieval_accuracy"))
            and float(zeva_validation["retrieval_accuracy"]) >= args.minimum_retrieval_accuracy,
            "aggregate_improvement": finite(zeva_validation.get("paired_improvement"))
            and float(zeva_validation["paired_improvement"]) > 0,
            "win_fraction": finite(zeva_validation.get("paired_win_fraction"))
            and float(zeva_validation["paired_win_fraction"]) >= args.minimum_win_fraction,
            "every_task_nonregression": finite(
                zeva_validation.get("minimum_task_paired_improvement")
            )
            and float(zeva_validation["minimum_task_paired_improvement"]) >= 0,
            "base_flow_finite": finite(base_validation.get("flow")),
            "zeva_flow_finite": finite(zeva_validation.get("flow")),
        }
        row = {
            "step": step,
            "base_checkpoint": str(base_paths[step]),
            "zeva_checkpoint": str(zeva_paths[step]),
            "base_validation_flow": base_validation.get("flow"),
            "zeva_validation_flow": zeva_validation.get("flow"),
            "zeva_residual_off_flow": zeva_validation.get("baseline"),
            "paired_improvement": zeva_validation.get("paired_improvement"),
            "paired_win_fraction": zeva_validation.get("paired_win_fraction"),
            "minimum_task_paired_improvement": zeva_validation.get(
                "minimum_task_paired_improvement"
            ),
            "retrieval_accuracy": zeva_validation.get("retrieval_accuracy"),
            "requirements": requirements,
            "errors": sorted(set(errors)),
        }
        row["eligible"] = not row["errors"] and all(requirements.values())
        rows.append(row)

    eligible = [row for row in rows if row["eligible"]]
    # The two branches must use one shared training duration.  First preserve
    # the strongest validation-only Base, then prefer safer ZeVA residuals.
    selected = min(
        eligible,
        key=lambda row: (
            float(row["base_validation_flow"]),
            -float(row["paired_improvement"]),
            -float(row["paired_win_fraction"]),
            float(row["zeva_validation_flow"]),
            int(row["step"]),
        ),
    ) if eligible else None

    selected_payload = None
    if selected is not None:
        base_checkpoint = Path(selected["base_checkpoint"])
        zeva_checkpoint = Path(selected["zeva_checkpoint"])
        selected_payload = {
            **selected,
            "artifacts": {
                "base_model_sha256": sha256(base_checkpoint / "model.safetensors"),
                "zeva_model_sha256": sha256(zeva_checkpoint / "model.safetensors"),
                "zeva_adapter_sha256": sha256(zeva_checkpoint / "zeva_adapter.pth"),
            },
        }

    report = {
        "schema": "zeva-robotwin-action-expert-pair-validation-selection-v1",
        "selection_data": "train95_validation5_only",
        "closed_loop_metrics_used": False,
        "shared_step_required": True,
        "selection_order": [
            "minimum_base_validation_flow",
            "maximum_zeva_paired_improvement",
            "maximum_zeva_paired_win_fraction",
            "minimum_zeva_validation_flow",
            "earlier_step",
        ],
        "eligible_count": len(eligible),
        "rows": rows,
        "selected": selected_payload,
    }
    destination = root / "deployment_selection.json"
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps({"eligible_count": len(eligible), "selected_step": (
        selected_payload or {}
    ).get("step"), "output": str(destination)}, indent=2))
    if selected_payload is None:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
