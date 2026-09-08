#!/usr/bin/env python3
"""Apply a preregistered global deployment scale to a trained ZeVA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--candidate-set", default="closed-loop-validation-grid-v1")
    args = parser.parse_args()
    if not math.isfinite(args.scale) or not 0.0 < args.scale <= 1.0:
        raise ValueError("Residual scale candidate must be finite and in (0, 1].")

    source = args.adapter.resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") not in {
        "zeva-robotwin-stage2-adapter-v5",
        "zeva-robotwin-pi05-adapter-v4",
    }:
        raise RuntimeError(f"Unsupported adapter schema: {checkpoint.get('schema')!r}")
    checkpoint["deployment_task_residual_scales"] = {}
    checkpoint["deployment_default_residual_scale"] = float(args.scale)
    checkpoint["deployment_residual_calibration"] = {
        "schema": "zeva-robotwin-residual-scale-candidate-v1",
        "candidate_set": args.candidate_set,
        "global_scale": float(args.scale),
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
            {"output": str(args.output), "scale": args.scale, "source_sha256": sha256(source)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
