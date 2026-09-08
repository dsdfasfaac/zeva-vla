#!/usr/bin/env python3
"""Interpolate a trained RoboTwin action expert toward its PI0.5 anchor.

Only the action path is allowed to differ.  Frozen PaliGemma tensors are copied
from the anchor after an exact equality check, while a ZeVA adapter can be
copied alongside the interpolated foundation unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ACTION_PREFIXES = (
    "model.paligemma_with_expert.gemma_expert.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.time_mlp_in.",
    "model.time_mlp_out.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_action_tensor(name: str) -> bool:
    return name.startswith(ACTION_PREFIXES)


def interpolate(anchor_path: Path, trained_path: Path, output_path: Path, alpha: float) -> dict:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    tensors: dict[str, torch.Tensor] = {}
    changed = 0
    frozen = 0
    max_frozen_difference = 0.0
    with safe_open(anchor_path, framework="pt", device="cpu") as anchor, safe_open(
        trained_path, framework="pt", device="cpu"
    ) as trained:
        anchor_keys = list(anchor.keys())
        trained_keys = list(trained.keys())
        if anchor_keys != trained_keys:
            raise ValueError("anchor and trained checkpoints have different tensor keys")
        for name in anchor_keys:
            base = anchor.get_tensor(name)
            tuned = trained.get_tensor(name)
            if base.shape != tuned.shape or base.dtype != tuned.dtype:
                raise ValueError(f"tensor contract differs for {name}")
            if _is_action_tensor(name):
                # Perform interpolation in fp32, then restore the released dtype.
                tensors[name] = torch.lerp(base.float(), tuned.float(), alpha).to(base.dtype)
                changed += 1
            else:
                if not torch.equal(base, tuned):
                    difference = float((base.float() - tuned.float()).abs().max())
                    max_frozen_difference = max(max_frozen_difference, difference)
                    raise ValueError(
                        f"frozen tensor changed in trained checkpoint: {name} (max abs {difference})"
                    )
                tensors[name] = base
                frozen += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp-{os.getpid()}")
    save_file(tensors, temporary)
    os.replace(temporary, output_path)
    return {
        "schema": "robotwin-action-expert-anchor-interpolation-v1",
        "alpha_trained": alpha,
        "alpha_anchor": 1.0 - alpha,
        "anchor_checkpoint": str(anchor_path.resolve()),
        "anchor_sha256": _sha256(anchor_path),
        "trained_checkpoint": str(trained_path.resolve()),
        "trained_sha256": _sha256(trained_path),
        "output_checkpoint": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "action_tensor_count": changed,
        "verified_frozen_tensor_count": frozen,
        "max_frozen_difference": max_frozen_difference,
        "selection_protocol": "alpha must be selected without closed-loop test outcomes",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    output_model = args.output_dir / "model.safetensors"
    manifest = interpolate(args.anchor, args.trained, output_model, args.alpha)
    if args.adapter is not None:
        if not args.adapter.is_file():
            raise FileNotFoundError(args.adapter)
        shutil.copy2(args.adapter, args.output_dir / "zeva_adapter.pth")
        manifest["adapter"] = str(args.adapter.resolve())
        manifest["adapter_sha256"] = _sha256(args.adapter)
        manifest["adapter_interpolated"] = False
    manifest_path = args.output_dir / "interpolation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
