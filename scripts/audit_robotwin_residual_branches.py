#!/usr/bin/env python3
"""Audit the relative strength of RoboTwin ZeVA residual branches.

This is a lightweight, foundation-free diagnostic.  It evaluates the trained
adapter on each selected task's mean training-language embedding and all causal
bank phase bins.  The result is not a closed-loop metric; it is intended to
detect collapsed routing or a branch whose injected signal is negligibly small
before spending simulator time on ablations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from openpi.zeva.causal_bank import RobotWinCausalBank
from openpi.zeva.context import MemoryContextEncoder
from openpi.zeva.robotwin_policy import RobotWinActionPrior


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _rms(values: torch.Tensor) -> float:
    return float(values.detach().float().square().mean().sqrt().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--goal-embeddings", type=Path, required=True)
    parser.add_argument("--causal-bank", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    adapter = torch.load(args.adapter.resolve(), map_location="cpu", weights_only=False)
    if adapter.get("schema") not in {
        "zeva-robotwin-stage2-adapter-v5",
        "zeva-robotwin-pi05-adapter-v4",
    }:
        raise RuntimeError(f"Unsupported adapter schema: {adapter.get('schema')!r}")
    language = torch.load(
        args.goal_embeddings.resolve(), map_location="cpu", weights_only=False
    )
    split = language["splits"]["train"]
    record_ids = list(split["record_ids"])
    embeddings = split["embeddings"].float()
    bank = RobotWinCausalBank.load(args.causal_bank.resolve(), device="cpu")
    task_names = tuple(_load_json(args.task_manifest.resolve())["task_names"])

    task_projector = nn.Sequential(nn.Linear(2048, 256), nn.LayerNorm(256))
    task_projector.load_state_dict(adapter["task_token_projector"])
    context_encoder = MemoryContextEncoder()
    context_encoder.load_state_dict(adapter["memory_context_encoder"])
    action_prior = RobotWinActionPrior(task_dim=256, phase_dim=128, context_dim=256)
    action_prior.load_state_dict(adapter["action_prior"])
    context_projector = nn.Linear(256, 1024)
    context_projector.load_state_dict(adapter["causal_action_projector"])
    prior_projector = nn.Linear(16, 1024)
    prior_projector.load_state_dict(adapter["prior_action_projector"])
    router = nn.Sequential(nn.Linear(384, 2))
    router.load_state_dict(adapter["residual_gate_router"])
    modules = (
        task_projector,
        context_encoder,
        action_prior,
        context_projector,
        prior_projector,
        router,
    )
    for module in modules:
        module.eval()

    task_rows: dict[str, Any] = {}
    all_context_gates = []
    all_prior_gates = []
    all_context_residuals = []
    all_prior_residuals = []
    with torch.no_grad():
        for task_name in task_names:
            try:
                task_id = bank.task_names.index(task_name)
            except ValueError as exc:
                raise RuntimeError(f"Task is absent from causal bank: {task_name}") from exc
            indices = [
                index
                for index, record_id in enumerate(record_ids)
                if record_id.split(":")[1] == task_name
            ]
            if not indices:
                raise RuntimeError(f"Task is absent from language table: {task_name}")
            phase = bank.phase_key[task_id].float()
            mean_language = embeddings[torch.tensor(indices)].mean(dim=0)
            task = task_projector(mean_language.repeat(len(phase), 1))
            task_ids = torch.full((len(phase),), task_id, dtype=torch.long)
            memory = bank.retrieve(task_ids, phase, brief_size=8, retrieval_top_k=5)
            context = context_encoder(
                task,
                phase,
                memory.brief_signals.float(),
                memory.retrieved_signals.float(),
                memory.brief_mask,
                memory.retrieved_mask,
            )
            prior_mean = action_prior(task, phase, context).mean
            multipliers = 2.0 * torch.sigmoid(router(torch.cat([task, phase], dim=-1)))
            context_gate = torch.sigmoid(adapter["context_gate_logit"]) * multipliers[:, 0]
            prior_gate = torch.sigmoid(adapter["prior_gate_logit"]) * multipliers[:, 1]
            context_residual = context_projector(context) * context_gate[:, None]
            prior_residual = prior_projector(prior_mean) * prior_gate[:, None, None]
            task_rows[task_name] = {
                "language_records": len(indices),
                "phase_bins": len(phase),
                "context_gate": _summary(context_gate),
                "prior_gate": _summary(prior_gate),
                "injected_context_residual_rms": _rms(context_residual),
                "injected_prior_residual_rms": _rms(prior_residual),
                "prior_mean_rms": _rms(prior_mean),
            }
            all_context_gates.append(context_gate)
            all_prior_gates.append(prior_gate)
            all_context_residuals.append(context_residual.flatten())
            all_prior_residuals.append(prior_residual.flatten())

    context_rms = _rms(torch.cat(all_context_residuals))
    prior_rms = _rms(torch.cat(all_prior_residuals))
    report = {
        "schema": "zeva-robotwin-residual-branch-audit-v1",
        "scope": (
            "foundation-free prototype diagnostic over mean train-language embedding "
            "and every causal-bank phase bin; not a closed-loop success metric"
        ),
        "adapter": str(args.adapter.resolve()),
        "goal_embeddings": str(args.goal_embeddings.resolve()),
        "causal_bank": str(args.causal_bank.resolve()),
        "task_manifest": str(args.task_manifest.resolve()),
        "aggregate": {
            "context_gate": _summary(torch.cat(all_context_gates)),
            "prior_gate": _summary(torch.cat(all_prior_gates)),
            "injected_context_residual_rms": context_rms,
            "injected_prior_residual_rms": prior_rms,
            "context_to_prior_injected_rms_ratio": context_rms / max(prior_rms, 1e-12),
        },
        "tasks": task_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
