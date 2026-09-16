#!/usr/bin/env python3
"""Select the earliest Base-locked H15 candidate using validation only.

No RoboTwin rollout outcomes, success labels, or task-level test metrics enter
this selector.  Reports are produced by the read-only Stage2 diagnostics path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_STEPS = (500, 1000)
EXPECTED_BASE_SHA = "bcf1d4f5f3e77926b8b7f798bb983378ad096e3892eed1a66058c5d98ed9bc17"
EXPECTED_SAMPLE_SHA = "a9c6a8e30aa3f7ecbfcb9ce81d6a141ef563300d41e339a22d06fb75a0418ac1"
EXPECTED_ADAPTER_SHA = "8ac54abcec7704b0111b7c28be3fb3a18e27e0ebe8e3dcf8ddff948b36100f8f"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_mean(report: dict[str, Any], key: str) -> float:
    cell = report["result"]["validation_diagnostics"]["flow"][key]
    if cell.get("available") is not True or cell.get("valid_examples") != 5874:
        raise ValueError(f"Incomplete {key} H15 diagnostic")
    value = float(cell["mean"])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} H15 diagnostic")
    return value


def _validated_row(step: int, path: Path) -> dict[str, Any]:
    if step not in EXPECTED_STEPS:
        raise ValueError(f"Unexpected checkpoint step {step}")
    report = json.loads(path.read_text(encoding="utf-8"))
    protocol = report["protocol"]
    expected_protocol = {
        "complete": True,
        "batch_size": 16,
        "seed": 1000,
        "validation_split": "validation5",
        "validation_decision_samples": 5874,
        "available_validation_batches": 368,
        "evaluated_batches": 368,
        "ordered_validation_samples_sha256": EXPECTED_SAMPLE_SHA,
        "policy_horizon": 50,
        "executed_horizon": 15,
        "optimizer_created": False,
        "checkpoint_written": False,
        "training_variant": "zeva",
        "video_backend": "torchcodec",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"Step {step}: protocol {key}={protocol.get(key)!r}, expected {expected!r}")
    intervention = protocol.get("validation_only_intervention", {})
    if intervention != {
        "context_gate_scale": 1.0,
        "prior_gate_scale": 1.0,
        "deployment_or_checkpoint_modified": False,
    }:
        raise ValueError(f"Step {step}: validation intervention changed the deployed policy")
    checkpoint = report["checkpoint"]
    if Path(checkpoint["path"]).name != f"{step:06d}":
        raise ValueError(f"Step {step}: checkpoint path does not identify the declared step")
    if checkpoint.get("model_sha256") != EXPECTED_BASE_SHA:
        raise ValueError(f"Step {step}: Base action-path weights are not byte-identical")
    adapter = checkpoint["adapter"]
    if not adapter.get("sha256") or not Path(adapter["path"]).name == "zeva_adapter.pth":
        raise ValueError(f"Step {step}: ZeVA adapter identity is missing")
    teacher = report["fixed_teacher"]
    if teacher.get("available") is not True or teacher.get("model_sha256") != EXPECTED_BASE_SHA:
        raise ValueError(f"Step {step}: fixed trained Base identity mismatch")
    if teacher.get("shared_frozen_weights_verified") is not True:
        raise ValueError(f"Step {step}: Stage1/VLM frozen-weight verification missing")
    if report["lineage"].get("stage1_schema") != "zeva-robotwin-zte-stage1-v2-checkpoint":
        raise ValueError(f"Step {step}: selected Stage1 is not ZTE v2")
    if report["lineage"].get("transition_horizon") != 15:
        raise ValueError(f"Step {step}: recurrent phase horizon mismatch")
    if report["lineage"].get("causal_bank_sha256") != "33b11b41992d07890a798602cb8cadebc0f798b8959adfc3b8cf5b660bf14ad9":
        raise ValueError(f"Step {step}: frozen causal bank identity mismatch")
    if protocol.get("dataset_adapter_sha256") != EXPECTED_ADAPTER_SHA:
        raise ValueError(f"Step {step}: validation dataset adapter identity mismatch")
    source_manifest = report["source_manifest"]
    manifest_path = Path(source_manifest["path"])
    if manifest_path != Path(checkpoint["path"]).parent / "manifest.json":
        raise ValueError(f"Step {step}: source manifest is not beside the selected checkpoint")
    if not manifest_path.is_file() or _sha256(manifest_path) != source_manifest["sha256"]:
        raise ValueError(f"Step {step}: source training manifest is missing or checksum mismatch")
    train_args = json.loads(manifest_path.read_text(encoding="utf-8"))["train_args"]
    if not (
        train_args.get("zeva_h15_flow_objective") is True
        and train_args.get("decouple_action_expert_gradient") is True
        and float(train_args.get("action_expert_learning_rate", -1)) == 0.0
    ):
        raise ValueError(f"Step {step}: source training objective is not Base-locked H15")
    on = _finite_mean(report, "zeva_residual_on_executed_h15")
    off = _finite_mean(report, "current_residual_off_executed_h15")
    base_cell = report["result"]["validation_diagnostics"]["paired"]["fixed_teacher"]
    if base_cell.get("available") is not True or base_cell.get("valid_examples") != 5874:
        raise ValueError(f"Step {step}: fixed Base paired H15 diagnostic incomplete")
    base = float(base_cell["teacher_flow"])
    if not math.isfinite(base):
        raise ValueError(f"Step {step}: non-finite fixed Base H15 error")
    if not math.isclose(float(base_cell["student_flow"]), on, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"Step {step}: fixed Base comparison uses a different ZeVA value")
    return {
        "step": step,
        "report": str(path.resolve()),
        "report_sha256": _sha256(path),
        "checkpoint": checkpoint["path"],
        "model_sha256": checkpoint["model_sha256"],
        "adapter_sha256": adapter["sha256"],
        "sample_order_sha256": protocol["ordered_validation_samples_sha256"],
        "h15_on": on,
        "h15_off": off,
        "h15_base": base,
        "on_minus_off": on - off,
        "on_minus_base": on - base,
        "passes": on < off and on < base,
    }


def select(report_paths: dict[int, Path]) -> dict[str, Any]:
    if tuple(sorted(report_paths)) != EXPECTED_STEPS:
        raise ValueError(f"Both fixed steps {EXPECTED_STEPS} are required")
    rows = [_validated_row(step, report_paths[step]) for step in EXPECTED_STEPS]
    if not math.isclose(rows[0]["h15_base"], rows[1]["h15_base"], rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError("Fixed Base H15 value differs between same-noise reports")
    selected = next((row for row in rows if row["passes"]), None)
    return {
        "schema": "zeva-robotwin-h15-validation-selector-v1",
        "selection_split": "validation5",
        "test_success_labels_used": False,
        "decision": "selected" if selected else "rejected_offline",
        "selected_step": selected["step"] if selected else None,
        "selected_checkpoint": selected["checkpoint"] if selected else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step500-report", required=True, type=Path)
    parser.add_argument("--step1000-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = select({500: args.step500_report, 1000: args.step1000_report})
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing selection: {args.output}")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("decision", "selected_step", "selected_checkpoint")}))


if __name__ == "__main__":
    main()
