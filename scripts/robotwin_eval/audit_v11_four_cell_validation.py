#!/usr/bin/env python3
"""Audit the four independent seed x diffusion-RNG validation cells for v11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=8)
    parser.add_argument("--minimum-total-gain", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell_paths = {
        "seed15000-rng20260907": args.root / "split-i" / "paired_report.json",
        "seed16000-rng20260908": args.root / "split-j" / "paired_report.json",
        "seed15000-rng20260908": args.root / "cross" / "split-i" / "paired_report.json",
        "seed16000-rng20260907": args.root / "cross" / "split-j" / "paired_report.json",
    }
    reports = {name: json.loads(path.read_text()) for name, path in cell_paths.items()}
    expected_tasks: set[str] | None = None
    task_delta_counts: dict[str, int] = {}
    cells = []
    for name, report in reports.items():
        tasks = {row["task"] for row in report["tasks"]}
        if expected_tasks is None:
            expected_tasks = tasks
        if tasks != expected_tasks:
            raise RuntimeError(f"task set differs in {name}")
        expected_total = len(tasks) * args.episodes_per_task
        if report["total_paired_episodes"] != expected_total:
            raise RuntimeError(f"{name} has {report['total_paired_episodes']} episodes, expected {expected_total}")
        delta_count = round(report["absolute_delta"] * expected_total)
        cells.append(
            {
                "cell": name,
                "baseline_success_rate": report["baseline_success_rate"],
                "zeva_success_rate": report["zeva_success_rate"],
                "delta_count": delta_count,
                "paired_episodes": expected_total,
            }
        )
        for row in report["tasks"]:
            task_delta_counts[row["task"]] = task_delta_counts.get(row["task"], 0) + round(
                row["delta"] * args.episodes_per_task
            )

    total_gain = sum(cell["delta_count"] for cell in cells)
    total_episodes = sum(cell["paired_episodes"] for cell in cells)
    baseline_successes = sum(round(cell["baseline_success_rate"] * cell["paired_episodes"]) for cell in cells)
    zeva_successes = sum(round(cell["zeva_success_rate"] * cell["paired_episodes"]) for cell in cells)
    negative_cells = [cell["cell"] for cell in cells if cell["delta_count"] < 0]
    negative_tasks = sorted(task for task, delta in task_delta_counts.items() if delta < 0)
    accepted = not negative_cells and not negative_tasks and total_gain >= args.minimum_total_gain
    payload = {
        "schema": "zeva-robotwin-v11-four-cell-validation-v1",
        "accepted_for_seed1000_formal_evaluation": accepted,
        "criteria": {
            "each_cell_delta_count_at_least": 0,
            "each_task_aggregate_delta_count_at_least": 0,
            "minimum_total_gain": args.minimum_total_gain,
        },
        "cells": cells,
        "total_paired_episodes": total_episodes,
        "baseline_successes": baseline_successes,
        "zeva_successes": zeva_successes,
        "total_gain": total_gain,
        "task_delta_counts": dict(sorted(task_delta_counts.items())),
        "negative_cells": negative_cells,
        "negative_tasks": negative_tasks,
    }
    destination = args.root / "four_cell_validation_summary.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not accepted:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
