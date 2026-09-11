"""Select an immutable scheduler-repaired Stage-1 checkpoint.

Selection is deliberately held-out-only: among the complete validation
snapshots from the repaired run, choose the minimum ``validation_loss``.  The
script never consults ``zte_v2_best.pth``/``zte_v2_latest.pth`` or any
closed-loop metric, and records every candidate score plus the selected file's
SHA256 for downstream artifact provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch


EXPECTED_SCHEMA = "zeva-robotwin-zte-stage1-v2-checkpoint"
EXPECTED_VALIDATION_EPISODES = 1350
CHECKPOINT_INTERVAL = 512

# These fields define the representation/objective contract.  Optimizer
# schedule metadata is recorded separately; it must not make two checkpoints
# from this repaired run incomparable merely because their step changed.
OBJECTIVE_KEYS = (
    "executed_action_steps",
    "transition_stride",
    "effect_steps",
    "action_prediction_context",
    "prediction_loss_reduction",
    "task_paired_batches",
    "use_effect_stream",
    "use_mamba",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pid_active(pid: int) -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if not proc_stat.exists():
        return False
    try:
        fields = proc_stat.read_text().split()
    except OSError:
        return False
    # Linux /proc/<pid>/stat field 3 is the process state.  A zombie has
    # exited; the waiter only proceeds once the launcher is no longer live.
    return len(fields) < 3 or fields[2] != "Z"


def _objective_signature(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("zte_config", {})
    args = payload.get("manifest", {}).get("train_args", {})
    return {key: config.get(key, args.get(key)) for key in OBJECTIVE_KEYS}


def _check_log(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_supplied", "path": None}
    if not path.exists():
        return {"status": "unavailable", "path": str(path)}
    text = path.read_text(errors="replace")
    failure_patterns = (
        r"Traceback \(most recent call last\)",
        r"ChildFailedError",
        r"CUDA out of memory",
        r"NCCL.*(?:error|fail)",
        r"RuntimeError:",
        r"\bSIG(?:KILL|TERM|SEGV)\b",
    )
    failures = [pattern for pattern in failure_patterns if re.search(pattern, text, re.IGNORECASE)]
    return {
        "status": "failure_detected" if failures else "clean",
        "path": str(path),
        "failure_patterns": failures,
        "bytes": path.stat().st_size,
    }


def select_checkpoint(
    run_dir: Path,
    *,
    final_step: int,
    output: Path,
    training_pid: int | None = None,
    training_log: Path | None = None,
    allow_missing_log: bool = False,
) -> dict[str, Any]:
    if final_step <= 0 or final_step % CHECKPOINT_INTERVAL:
        raise ValueError("final_step must be a positive multiple of 512.")
    if training_pid is not None and _pid_active(training_pid):
        raise RuntimeError(f"training PID {training_pid} is still active; refusing selection.")
    log_status = _check_log(training_log)
    if log_status["status"] == "failure_detected":
        raise RuntimeError(f"training log reports failure: {training_log}")
    if log_status["status"] in {"not_supplied", "unavailable"} and not allow_missing_log:
        raise RuntimeError("A non-failing training log is required unless --allow-missing-log is set.")

    candidates: list[dict[str, Any]] = []
    objective_signature: dict[str, Any] | None = None
    for step in range(CHECKPOINT_INTERVAL * 3, final_step + 1, CHECKPOINT_INTERVAL):
        path = run_dir / f"zte_v2_step_{step:06d}.pth"
        if not path.exists():
            raise FileNotFoundError(f"Required complete-run checkpoint is missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != EXPECTED_SCHEMA:
            raise ValueError(f"Checkpoint {path} has unexpected schema.")
        if int(payload.get("step", -1)) != step:
            raise ValueError(f"Checkpoint {path} declares a different step.")
        validation = payload.get("validation", {})
        complete = bool(validation.get("validation_complete"))
        episodes = int(validation.get("validation_episode_count", -1))
        loss = float(validation.get("validation_loss", math.nan))
        if not complete or episodes != EXPECTED_VALIDATION_EPISODES or not math.isfinite(loss):
            raise ValueError(
                f"Checkpoint {path} lacks complete held-out validation: "
                f"complete={complete}, episodes={episodes}, loss={loss}."
            )
        signature = _objective_signature(payload)
        if objective_signature is None:
            objective_signature = signature
        elif signature != objective_signature:
            raise ValueError(
                f"Checkpoint {path} objective signature differs from the repaired run: "
                f"{signature} != {objective_signature}"
            )
        candidates.append(
            {
                "step": step,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "validation_loss": loss,
                "validation_episode_count": episodes,
                "validation_complete": complete,
                "task_probe_accuracy": float(validation.get("task_probe_accuracy", math.nan)),
                "phase_order_accuracy": float(validation.get("phase_order_accuracy", math.nan)),
                "effect_cosine": float(validation.get("effect_cosine", math.nan)),
            }
        )
        del payload

    selected = min(candidates, key=lambda item: (item["validation_loss"], item["step"]))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint selection: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "zeva-robotwin-zte-scheduler-repaired-selection-v1",
        "run_dir": str(run_dir.resolve()),
        "final_step_required": final_step,
        "training_pid": training_pid,
        "training_log": log_status,
        "objective_signature": objective_signature,
        "candidates": candidates,
        "selected": selected,
        "selection_reason": (
            "minimum complete validation_loss over repaired-run checkpoints; "
            "held-out validation only, no closed-loop metric or mutable best/latest alias"
        ),
        "immutable_pin": {
            "filename": selected["path"],
            "sha256": selected["sha256"],
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-step", type=int, default=4096)
    parser.add_argument("--training-pid", type=int)
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--allow-missing-log", action="store_true")
    args = parser.parse_args()
    result = select_checkpoint(
        args.run_dir,
        final_step=args.final_step,
        output=args.output,
        training_pid=args.training_pid,
        training_log=args.training_log,
        allow_missing_log=args.allow_missing_log,
    )
    print(json.dumps({"selected": result["selected"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
