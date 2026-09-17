"""Apply the preregistered validation5-only gate to the new H15 residual run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def select(plan: dict, run_root: Path) -> dict:
    thresholds = plan["validation_gate_numeric"]
    expected_count = int(thresholds["expected_decisions"])
    required_gain = float(thresholds["minimum_relative_h15_mse_improvement"])
    required_tasks = int(thresholds["minimum_nonworse_tasks"])
    if expected_count <= 0 or not 0 < required_gain < 1 or not 0 < required_tasks <= 10:
        raise ValueError("Invalid preregistered validation gate")
    expected_tasks = 10
    candidates = []
    for step in sorted(int(x) for x in plan["checkpoint_steps"]):
        checkpoint = run_root / f"{step:06d}"
        state_file = checkpoint / "training_state.pt"
        model_file = checkpoint / "model.safetensors"
        adapter_file = checkpoint / "zeva_adapter.pth"
        # training_state.pt is a torch pickle. A small local import is used
        # here rather than embedding torch in the caller's orchestration.
        if not all(path.is_file() and path.stat().st_size > 0 for path in (
            state_file, model_file, adapter_file,
        )):
            candidates.append({"step": step, "passed": False, "reason": "missing_artifact"})
            continue
        import torch  # noqa: PLC0415

        state = torch.load(state_file, map_location="cpu", weights_only=False)
        manifest = state["manifest"]
        if manifest["train_args"]["training_variant"] != "output_residual":
            raise ValueError(f"Unexpected training variant at step {step}")
        if manifest["initial_stage2_identity"]["model_sha256"] != plan["initial_base_model_sha256"]:
            raise ValueError(f"Base1000 checkpoint identity drift at step {step}")
        validation = state["validation"]
        by_task = validation["per_task_paired"]
        if len(by_task) != expected_tasks:
            raise ValueError(f"Expected {expected_tasks} tasks, got {len(by_task)}")
        total_count = sum(int(row["count"]) for row in by_task.values())
        if total_count != expected_count:
            raise ValueError(f"Validation coverage {total_count} != {expected_count}")
        base = sum(float(row["base_h15_mse"]) * int(row["count"]) for row in by_task.values()) / total_count
        corrected = sum(float(row["corrected_h15_mse"]) * int(row["count"]) for row in by_task.values()) / total_count
        improvement = (base - corrected) / base if base > 0 else float("nan")
        nonworse = sum(
            float(row["corrected_h15_mse"]) <= float(row["base_h15_mse"])
            for row in by_task.values()
        )
        finite = all(math.isfinite(float(validation[key])) for key in (
            "residual_abs", "prior_residual_keep", "gate", "retrieval_accuracy",
        )) and all(math.isfinite(value) for value in (base, corrected, improvement))
        passed = finite and improvement >= required_gain and nonworse >= required_tasks
        candidates.append({
            "step": step,
            "checkpoint": str(checkpoint),
            "passed": passed,
            "validation_decisions": total_count,
            "base_h15_mse": base,
            "corrected_h15_mse": corrected,
            "relative_h15_mse_improvement": improvement,
            "nonworse_tasks": nonworse,
            "finite_diagnostics": finite,
        })
    winner = next((candidate for candidate in candidates if candidate["passed"]), None)
    return {
        "schema": "zeva-base1000-ztev2-validation5-selection-v1",
        "formal_test_success_labels_used": False,
        "selected_checkpoint": winner["checkpoint"] if winner else None,
        "selected_step": winner["step"] if winner else None,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    result = select(plan, args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
