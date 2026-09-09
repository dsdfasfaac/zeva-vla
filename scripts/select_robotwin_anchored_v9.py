#!/usr/bin/env python3
"""Select an anchored-v9 ZeVA checkpoint using train95/validation5 only."""

from __future__ import annotations

import argparse
import gc
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


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def checkpoints(root: Path) -> dict[int, Path]:
    result = {
        int(path.name): path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and (path / "training_state.pt").is_file()
        and (path / "model.safetensors").is_file()
        and (path / "zeva_adapter.pth").is_file()
    }
    if not result:
        raise RuntimeError(f"No complete ZeVA checkpoints found under {root}")
    return result


def load_training_metadata(path: Path) -> dict[str, Any]:
    """Read scalar/dict metadata without materializing multi-GB optimizer tensors."""
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError:  # pragma: no cover - compatibility with older PyTorch.
        state = torch.load(path, map_location="cpu", weights_only=False)
    else:
        with FakeTensorMode():
            state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = {
        key: state.get(key)
        for key in ("schema", "step", "manifest", "validation")
    }
    del state
    gc.collect()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("base_checkpoint", type=Path)
    parser.add_argument("--minimum-retrieval-accuracy", type=float, default=0.95)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.5)
    args = parser.parse_args()

    candidate_root = args.candidate_root.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    base_state = load_training_metadata(base_checkpoint / "training_state.pt")
    if base_state.get("schema") != "robotwin-pi05-action-expert-baseline-training-state-v1":
        raise ValueError("Anchored v9 requires the matched action-expert Base schema.")
    base_step = int(base_state.get("step", -1))
    if base_step != int(base_checkpoint.name):
        raise ValueError("Base checkpoint directory and training-state step differ.")
    base_manifest = base_state.get("manifest", {})
    base_manifest_sha256 = sha256(base_checkpoint.parent / "manifest.json")

    rows: list[dict[str, Any]] = []
    for step, path in sorted(checkpoints(candidate_root).items()):
        state = load_training_metadata(path / "training_state.pt")
        manifest = state.get("manifest", {})
        validation = state.get("validation", {})
        errors: list[str] = []
        variant = manifest.get("training_variant")
        if variant not in {"zeva", "adapter"}:
            errors.append("training_variant")
        expected_schema = {
            "zeva": "zeva-robotwin-stage2-action-expert-training-state-v8",
            "adapter": "zeva-robotwin-stage2-frozen-foundation-training-state-v2",
        }.get(variant)
        if state.get("schema") != expected_schema:
            errors.append("state_schema")
        if int(state.get("step", -1)) != step:
            errors.append("step_mismatch")
        if manifest.get("effective_global_batch_size") != 256:
            errors.append("global_batch")
        if manifest.get("foundation_identity", {}).get("model_sha256") != FOUNDATION_SHA256:
            errors.append("foundation_identity")
        if manifest.get("task_scope") != base_manifest.get("task_scope"):
            errors.append("task_scope")
        if len(manifest.get("task_scope", {}).get("task_names", ())) != TASK_COUNT:
            errors.append("task_count")
        initial = manifest.get("initial_stage2_identity", {})
        if (
            Path(initial.get("path", "/")).resolve() != base_checkpoint
            or int(initial.get("step", -1)) != base_step
            or initial.get("source_manifest_sha256") != base_manifest_sha256
        ):
            errors.append("initial_base_identity")
        preservation = manifest.get("paired_baseline_preservation", {})
        expected_teacher = (
            "independent_frozen_base_action_path"
            if variant == "zeva"
            else "current_student_residual_off"
        )
        if preservation.get("teacher") != expected_teacher:
            errors.append("paired_teacher")
        if variant == "zeva":
            anchor = manifest.get("anchor_stage2_identity", {})
            if (
                Path(anchor.get("path", "/")).resolve() != base_checkpoint
                or int(anchor.get("step", -1)) != base_step
                or anchor.get("source_manifest_sha256") != base_manifest_sha256
            ):
                errors.append("independent_anchor_identity")
        frozen = set(manifest.get("frozen", ()))
        required_frozen = {
            "pi05_paligemma_vision_tower",
            "pi05_paligemma_language_backbone",
            "zte",
            "causal_bank",
            "task_retrieval",
        }
        if not required_frozen.issubset(frozen):
            errors.append("frozen_contract")
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
            "every_task_nonregression": finite(
                validation.get("minimum_task_paired_improvement")
            )
            and float(validation["minimum_task_paired_improvement"]) >= 0,
            "base_teacher_flow_finite": finite(validation.get("baseline")),
            "zeva_flow_finite": finite(validation.get("flow")),
        }
        row = {
            "step": step,
            "base_step": base_step,
            "base_checkpoint": str(base_checkpoint),
            "zeva_checkpoint": str(path),
            "training_variant": variant,
            "base_validation_flow": validation.get("baseline"),
            "zeva_validation_flow": validation.get("flow"),
            "paired_improvement": validation.get("paired_improvement"),
            "paired_win_fraction": validation.get("paired_win_fraction"),
            "minimum_task_paired_improvement": validation.get(
                "minimum_task_paired_improvement"
            ),
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
        "schema": "zeva-robotwin-anchored-v9-validation-selection-v1",
        "selection_data": "train95_validation5_only",
        "closed_loop_metrics_used": False,
        "fixed_base_checkpoint": str(base_checkpoint),
        "base_selected_by": "minimum validation5 flow among completed v8 Base checkpoints",
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
