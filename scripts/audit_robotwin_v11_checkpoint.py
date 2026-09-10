#!/usr/bin/env python3
"""Audit v11 foundation drift and hard prior-only deployment invariants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from safetensors import safe_open
import torch


ACTION_PATH_MARKERS = (
    ".gemma_expert.model.",
    ".action_in_proj.",
    ".action_out_proj.",
    ".time_mlp_in.",
    ".time_mlp_out.",
)


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
    hard_invariants = {
        "adapter_schema_v5": adapter.get("schema") == "zeva-robotwin-stage2-adapter-v5",
        "training_variant_prior_zeva": manifest.get("training_variant") == "prior_zeva",
        "h15_prior_injection": adapter.get("prior_injection_horizon") == 15,
        "h15_prior_supervision": (manifest.get("action_prior") or {}).get(
            "supervision_horizon"
        )
        == 15,
        "direct_context_disabled": adapter.get("direct_context_injection_enabled") is False,
        "direct_context_projector_zero": _all_zero(adapter["causal_action_projector"]),
        "gate_router_identity_scaled": _all_zero(adapter["residual_gate_router"]),
        "prior_scalar_gate_half": math.isclose(
            float(torch.sigmoid(adapter["prior_gate_logit"])), 0.5, abs_tol=1e-7
        ),
    }

    changed_action_tensors = 0
    unchanged_action_tensors = 0
    frozen_tensors = 0
    frozen_changed: list[str] = []
    action_squared_difference = 0.0
    action_squared_reference = 0.0
    action_max_absolute_difference = 0.0
    with safe_open(args.base_model.resolve(), framework="pt", device="cpu") as base, safe_open(
        args.candidate_model.resolve(), framework="pt", device="cpu"
    ) as candidate:
        base_keys = set(base.keys())
        candidate_keys = set(candidate.keys())
        key_sets_identical = base_keys == candidate_keys
        for name in sorted(base_keys & candidate_keys):
            reference = base.get_tensor(name)
            value = candidate.get_tensor(name)
            is_action = any(marker in name for marker in ACTION_PATH_MARKERS)
            if not is_action:
                frozen_tensors += 1
                if not torch.equal(reference, value):
                    frozen_changed.append(name)
                continue
            difference = value.float() - reference.float()
            changed = torch.count_nonzero(difference).item() > 0
            changed_action_tensors += int(changed)
            unchanged_action_tensors += int(not changed)
            action_squared_difference += float(difference.double().square().sum())
            action_squared_reference += float(reference.double().square().sum())
            action_max_absolute_difference = max(
                action_max_absolute_difference, float(difference.abs().max())
            )

    hard_invariants["model_key_sets_identical"] = key_sets_identical
    hard_invariants["all_non_action_tensors_bit_identical"] = not frozen_changed
    passed = all(hard_invariants.values())
    report = {
        "schema": "zeva-robotwin-v11-checkpoint-audit-v1",
        "passed": passed,
        "base_model": str(args.base_model.resolve()),
        "candidate_model": str(args.candidate_model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "hard_invariants": hard_invariants,
        "foundation_drift": {
            "frozen_tensor_count": frozen_tensors,
            "frozen_changed_count": len(frozen_changed),
            "frozen_changed": frozen_changed,
            "changed_action_tensor_count": changed_action_tensors,
            "unchanged_action_tensor_count": unchanged_action_tensors,
            "action_relative_l2": math.sqrt(action_squared_difference)
            / max(math.sqrt(action_squared_reference), 1e-30),
            "action_max_absolute_difference": action_max_absolute_difference,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
