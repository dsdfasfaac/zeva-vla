#!/usr/bin/env python3
"""Audit v14's immutable PI0.5 and direct H15 residual deployment contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from safetensors import safe_open
import torch


def _all_zero(state: dict[str, torch.Tensor]) -> bool:
    return all(torch.count_nonzero(value).item() == 0 for value in state.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    adapter = torch.load(args.adapter.resolve(), map_location="cpu", weights_only=False)
    manifest = adapter.get("manifest") or {}
    residual_state = adapter.get("output_residual_corrector") or {}
    delta_state = {
        name: value for name, value in residual_state.items() if name.startswith("delta_head.")
    }
    hard_invariants = {
        "adapter_schema_v5": adapter.get("schema") == "zeva-robotwin-stage2-adapter-v5",
        "training_variant_output_residual": manifest.get("training_variant") == "output_residual",
        "output_residual_enabled": adapter.get("output_residual_correction_enabled") is True,
        "output_lerp_disabled": adapter.get("output_action_correction_enabled") is False,
        "direct_token_context_disabled": adapter.get("direct_context_injection_enabled") is False,
        "h15_output_residual": adapter.get("output_residual_horizon") == 15,
        "finite_positive_residual_bound": math.isfinite(
            float(adapter.get("output_residual_bound", float("nan")))
        )
        and float(adapter.get("output_residual_bound")) > 0,
        "legacy_context_projector_zero": _all_zero(adapter["causal_action_projector"]),
        "legacy_prior_projector_zero": _all_zero(adapter["prior_action_projector"]),
        "trained_residual_head_nonzero": bool(delta_state) and not _all_zero(delta_state),
        "manifest_h35_exact_base": (manifest.get("output_residual_correction") or {}).get("h35")
        == "exact_base_copy",
        "manifest_token_injection_false": (manifest.get("output_residual_correction") or {}).get(
            "token_injection"
        )
        is False,
    }

    changed: list[str] = []
    compared = 0
    with safe_open(args.base_model.resolve(), framework="pt", device="cpu") as base, safe_open(
        args.candidate_model.resolve(), framework="pt", device="cpu"
    ) as candidate:
        base_keys = set(base.keys())
        candidate_keys = set(candidate.keys())
        hard_invariants["model_key_sets_identical"] = base_keys == candidate_keys
        for name in sorted(base_keys & candidate_keys):
            compared += 1
            if not torch.equal(base.get_tensor(name), candidate.get_tensor(name)):
                changed.append(name)
    hard_invariants["all_foundation_tensors_bit_identical"] = not changed

    passed = all(hard_invariants.values())
    report = {
        "schema": "zeva-robotwin-v14-checkpoint-audit-v1",
        "passed": passed,
        "base_model": str(args.base_model.resolve()),
        "candidate_model": str(args.candidate_model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "hard_invariants": hard_invariants,
        "foundation_tensor_audit": {
            "compared_count": compared,
            "changed_count": len(changed),
            "changed": changed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
