#!/usr/bin/env python3
"""Select binary v14 task scales on one closed-loop development split.

The rule is deliberately fixed before the independent confirmation split:
enable a task only when ZeVA has at least one extra success and strictly more
Base-only/ZeVA-only discordant wins.  Every other task uses scale zero, which
is the construction-level exact PI0.5 fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _progress(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("complete") is not True:
        raise ValueError(f"incomplete progress file: {path}")
    if payload.get("completed_episodes") != payload["identity"].get("target_episodes"):
        raise ValueError(f"episode count mismatch: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-progress", type=Path, required=True)
    parser.add_argument("--zeva-progress", type=Path, required=True)
    parser.add_argument("--input-adapter", type=Path, required=True)
    parser.add_argument("--output-adapter", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    args = parser.parse_args()

    base_files = {path.name: path for path in args.base_progress.glob("*.json")}
    zeva_files = {path.name: path for path in args.zeva_progress.glob("*.json")}
    if not base_files or base_files.keys() != zeva_files.keys():
        raise ValueError("Base and ZeVA progress task sets must be identical and non-empty.")

    rows = []
    scales: dict[str, float] = {}
    for name in sorted(base_files):
        base = _progress(base_files[name])
        zeva = _progress(zeva_files[name])
        base_rows = {int(row["seed"]): row for row in base["episode_results"]}
        zeva_rows = {int(row["seed"]): row for row in zeva["episode_results"]}
        if base_rows.keys() != zeva_rows.keys():
            raise ValueError(f"paired seed mismatch for {name}")
        for seed in base_rows:
            if base_rows[seed].get("instruction") != zeva_rows[seed].get("instruction"):
                raise ValueError(f"paired instruction mismatch for {name} seed {seed}")
        base_successes = sum(bool(row["success"]) for row in base_rows.values())
        zeva_successes = sum(bool(row["success"]) for row in zeva_rows.values())
        wins = sum(
            bool(zeva_rows[seed]["success"]) and not bool(base_rows[seed]["success"])
            for seed in base_rows
        )
        losses = sum(
            bool(base_rows[seed]["success"]) and not bool(zeva_rows[seed]["success"])
            for seed in base_rows
        )
        enabled = zeva_successes >= base_successes + 1 and wins > losses
        task = base["identity"]["task_name"]
        scales[task] = 1.0 if enabled else 0.0
        rows.append(
            {
                "task": task,
                "episodes": len(base_rows),
                "base_successes": base_successes,
                "zeva_successes": zeva_successes,
                "delta": zeva_successes - base_successes,
                "zeva_only_wins": wins,
                "base_only_losses": losses,
                "deployment_scale": scales[task],
            }
        )

    adapter = torch.load(args.input_adapter.resolve(), map_location="cpu", weights_only=False)
    if adapter.get("output_residual_correction_enabled") is not True:
        raise ValueError("Input adapter is not a v14 output-residual checkpoint.")
    calibration = {
        "schema": "zeva-robotwin-v14-binary-task-scale-calibration-v1",
        "development_split": args.split_name,
        "rule": "scale=1 iff task delta>=1 and zeva_only_wins>base_only_losses; else scale=0",
        "final_or_confirmation_data_used": False,
        "rows": rows,
    }
    adapter["deployment_task_residual_scales"] = scales
    adapter["deployment_default_residual_scale"] = 0.0
    adapter["deployment_residual_calibration"] = calibration
    args.output_adapter.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter, args.output_adapter)
    report = dict(calibration)
    report["task_scales"] = scales
    report["enabled_tasks"] = [task for task, scale in scales.items() if scale == 1.0]
    report["disabled_tasks"] = [task for task, scale in scales.items() if scale == 0.0]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
