"""Measure the offline H15 oracle upper bound of v15 PI candidates.

The probe only compares cached actions with cached normalized expert targets.
It does not select candidates online, train a ranker, or claim executable
closed-loop performance.  A task-language Gaussian-prior selector is not run
here because loading a Stage1 bank and recurrent features would change this
architecture-free oracle measurement; the result records that limitation
explicitly.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
import tyro


@dataclasses.dataclass
class Args:
    cache: str = (
        "/mnt/100T/users/dingxin/VLA/zeva-runs/robotwin-v5-h15-tasklang/"
        "advantage10-pi-candidates-v15/pi_candidates_k4.pt"
    )
    output: str | None = None
    horizon: int = 15


def _mean(values: torch.Tensor) -> float:
    return float(values.mean().item()) if values.numel() else float("nan")


def _split_metrics(
    split: dict[str, torch.Tensor],
    task_names: list[str],
    *,
    horizon: int,
) -> dict[str, Any]:
    candidates = split["candidate_actions"].float()
    targets = split["normalized_expert_actions"].float()
    task_ids = split["task_ids"].long()
    sample_indices = split["sample_indices"].long()
    if candidates.ndim != 4 or targets.ndim != 3:
        raise ValueError(
            "Candidate cache must contain [N,K,H,16] candidates and [N,H,16] targets; "
            f"got {tuple(candidates.shape)} and {tuple(targets.shape)}."
        )
    if candidates.shape[0] != targets.shape[0] or candidates.shape[0] != task_ids.numel():
        raise ValueError(f"{len(task_ids)} task ids do not match {candidates.shape[0]} samples.")
    if candidates.shape[-2:] != targets.shape[-2:] or candidates.shape[-1] != 16:
        raise ValueError("Candidate/target action dimensions are not H50 EEF16-aligned.")
    if not 1 <= horizon <= candidates.shape[-2]:
        raise ValueError(f"horizon must be in [1,{candidates.shape[-2]}], got {horizon}.")
    if candidates.shape[1] < 2:
        raise ValueError("Oracle comparison requires at least two candidates.")
    if sample_indices.numel() != sample_indices.unique().numel():
        raise ValueError("Candidate cache contains duplicate sample indices.")
    if task_ids.numel() and int(task_ids.min()) < 0:
        raise ValueError("Candidate cache contains a negative task id.")
    if task_ids.numel() and int(task_ids.max()) >= len(task_names):
        raise ValueError("Candidate cache task id exceeds task_names table.")

    candidate_errors = (
        candidates[:, :, :horizon] - targets[:, None, :horizon]
    ).square().mean(dim=(2, 3))
    base_errors = candidate_errors[:, 0]
    oracle_errors, oracle_indices = candidate_errors.min(dim=1)
    base_mean = base_errors.mean()
    oracle_mean = oracle_errors.mean()
    absolute_improvement = base_mean - oracle_mean
    relative_improvement = absolute_improvement / base_mean.clamp_min(1e-12)
    sample_relative = (base_errors - oracle_errors) / base_errors.clamp_min(1e-12)
    nonbase = oracle_indices != 0

    # Diversity is diagnostic only: it says whether independent draws differ,
    # not whether any particular draw is safe to execute.
    base_to_other = (candidates[:, 1:, :horizon] - candidates[:, :1, :horizon]).square().mean(
        dim=(2, 3)
    )
    pairwise = []
    for left in range(candidates.shape[1]):
        for right in range(left + 1, candidates.shape[1]):
            pairwise.append(
                (candidates[:, left, :horizon] - candidates[:, right, :horizon])
                .square()
                .mean(dim=(1, 2))
            )
    pairwise_mean = torch.stack(pairwise, dim=1).mean()

    per_task: dict[str, Any] = {}
    for task_id, task_name in enumerate(task_names):
        mask = task_ids == task_id
        if not bool(mask.any()):
            continue
        task_base = base_errors[mask]
        task_oracle = oracle_errors[mask]
        task_abs = task_base - task_oracle
        task_nonbase = nonbase[mask]
        task_base_mean = task_base.mean()
        per_task[task_name] = {
            "task_id": task_id,
            "count": int(mask.sum()),
            "candidate0_base_mse_h15": _mean(task_base),
            "oracle_min_k_mse_h15": _mean(task_oracle),
            "oracle_absolute_improvement": _mean(task_abs),
            "oracle_relative_improvement": float(
                (task_base_mean - task_oracle.mean()).div(task_base_mean.clamp_min(1e-12))
            ),
            "oracle_selected_nonbase_fraction": float(task_nonbase.float().mean()),
            "oracle_selected_candidate_histogram": {
                str(index): int((oracle_indices[mask] == index).sum())
                for index in range(candidates.shape[1])
            },
        }

    return {
        "count": int(candidates.shape[0]),
        "candidate_count": int(candidates.shape[1]),
        "horizon": horizon,
        "candidate0_base_mse_h15": _mean(base_errors),
        "oracle_min_k_mse_h15": _mean(oracle_errors),
        "oracle_absolute_improvement": float(absolute_improvement),
        "oracle_relative_improvement": float(relative_improvement),
        "oracle_mean_sample_relative_improvement": _mean(sample_relative),
        "oracle_selected_nonbase_fraction": float(nonbase.float().mean()),
        "oracle_selected_candidate_histogram": {
            str(index): int((oracle_indices == index).sum())
            for index in range(candidates.shape[1])
        },
        "candidate0_to_nonbase_mean_h15_mse": _mean(base_to_other),
        "all_candidate_pairwise_mean_h15_mse": float(pairwise_mean),
        "per_task": per_task,
    }


def main(args: Args) -> None:
    cache_path = Path(args.cache).resolve()
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "zeva-robotwin-pi-candidate-cache-v15":
        raise ValueError(f"Unsupported v15 candidate cache schema: {payload.get('schema')!r}")
    if int(payload.get("candidate_count", 0)) < 2:
        raise ValueError("v15 candidate cache has fewer than two candidates.")
    task_names = list(payload.get("task_names", ()))
    if not task_names:
        raise ValueError("v15 candidate cache omitted task_names.")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("v15 candidate cache omitted split tensors.")

    result: dict[str, Any] = {
        "schema": "zeva-robotwin-pi-candidate-oracle-v15",
        "cache": str(cache_path),
        "cache_schema": payload["schema"],
        "candidate_count": int(payload["candidate_count"]),
        "task_names": task_names,
        "selected_task_names": payload.get("selected_task_names", task_names),
        "foundation": payload.get("foundation"),
        "goal_embedding_checkpoint": payload.get("goal_embedding_checkpoint"),
        "dataset_adapter": payload.get("dataset_adapter"),
        "dataset_root": payload.get("dataset_root"),
        "live_queries": payload.get("live_queries"),
        "task_subset": payload.get("task_subset"),
        "horizon": args.horizon,
        "oracle_definition": (
            "offline argmin over candidate H15 normalized action MSE against expert; "
            "not a deployable selector or closed-loop result"
        ),
        "splits": {},
        "stage1_gaussian_prior_selector": {
            "status": "not_run",
            "reason": (
                "This probe intentionally does not load the Stage1 bank/recurrent features; "
                "only architecture-free multi-PI oracle is measured."
            ),
        },
    }
    for split_name, split in splits.items():
        result["splits"][split_name] = _split_metrics(
            split,
            task_names,
            horizon=args.horizon,
        )

    output = Path(args.output).resolve() if args.output else cache_path.with_name(
        "pi_candidates_k4_oracle_v15.json"
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
