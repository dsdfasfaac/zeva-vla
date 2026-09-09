#!/usr/bin/env python3
"""Create a validation-only ZeVA candidate with explicit branch guidance.

The trained task/phase router is retained.  A disabled branch is made exactly
zero by clearing its projector, while an enabled branch receives an absolute
sigmoid gate probability.  This lets closed-loop validation distinguish the
context and Gaussian action-prior residuals without changing PI0.5 weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import torch

SUPPORTED_SCHEMAS = {
    "zeva-robotwin-stage2-adapter-v5",
    "zeva-robotwin-pi05-adapter-v4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_branch(
    checkpoint: dict,
    *,
    probability: float,
    gate_key: str,
    projector_key: str,
) -> None:
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("Branch gate probability must be finite and in [0, 1).")
    if probability == 0.0:
        projector = checkpoint[projector_key]
        checkpoint[projector_key] = {
            name: torch.zeros_like(value) for name, value in projector.items()
        }
        # The zero projector is the exact-off invariant.  Keep a finite logit
        # so checkpoint serialization and mixed-precision loading remain safe.
        checkpoint[gate_key] = torch.zeros_like(checkpoint[gate_key])
        return
    value = torch.tensor(probability, dtype=torch.float32)
    checkpoint[gate_key] = torch.logit(value).to(
        dtype=checkpoint[gate_key].dtype,
        device=checkpoint[gate_key].device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-gate-probability", type=float, required=True)
    parser.add_argument("--prior-gate-probability", type=float, required=True)
    parser.add_argument(
        "--candidate-set",
        default="closed-loop-branch-guidance-validation-v1",
    )
    args = parser.parse_args()

    source = args.adapter.resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") not in SUPPORTED_SCHEMAS:
        raise RuntimeError(f"Unsupported adapter schema: {checkpoint.get('schema')!r}")

    configure_branch(
        checkpoint,
        probability=args.context_gate_probability,
        gate_key="context_gate_logit",
        projector_key="causal_action_projector",
    )
    configure_branch(
        checkpoint,
        probability=args.prior_gate_probability,
        gate_key="prior_gate_logit",
        projector_key="prior_action_projector",
    )
    checkpoint["deployment_task_residual_scales"] = {}
    checkpoint["deployment_default_residual_scale"] = 1.0
    checkpoint["deployment_residual_calibration"] = {
        "schema": "zeva-robotwin-branch-guidance-candidate-v1",
        "candidate_set": args.candidate_set,
        "context_gate_probability": args.context_gate_probability,
        "prior_gate_probability": args.prior_gate_probability,
        "context_exactly_disabled": args.context_gate_probability == 0.0,
        "prior_exactly_disabled": args.prior_gate_probability == 0.0,
        "task_phase_router": "trained_router_retained",
        "source_adapter": str(source),
        "source_adapter_sha256": sha256(source),
        "selection_split": "closed_loop_validation_only",
        "test_metrics_used": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "context_gate_probability": args.context_gate_probability,
                "prior_gate_probability": args.prior_gate_probability,
                "source_sha256": sha256(source),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
