#!/usr/bin/env python3
"""Audit the initial two-cell v11 paired validation.

The input root must contain ``split-i`` and ``split-j``.  Each split must
contain one completed 10-task Base/ZeVA pair with eight episodes per task.
This audit re-derives success counts from the per-episode progress files,
checks exact task/seed/instruction pairing, and only then applies the initial
two-cell gate: neither cell may regress and the combined gain must be at
least 6/160.  Per-task aggregates are reported for diagnosis but are not a
four-cell selection gate.

The summary is replaced atomically, including on validation failure, so a
stale passing summary cannot survive a missing or malformed input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "zeva-robotwin-paired-condition-v1"
EXPECTED_SPLITS = ("split-i", "split-j")
EXPECTED_TASK_COUNT = 10
EXPECTED_EPISODES_PER_TASK = 8
EXPECTED_CELL_EPISODES = EXPECTED_TASK_COUNT * EXPECTED_EPISODES_PER_TASK
MINIMUM_TOTAL_GAIN = 6


class AuditError(Exception):
    """A validation error that should be included in the summary."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the initial two-cell v11 paired validation."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="step500 validation root containing split-i/ and split-j/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="summary path (default: ROOT/validation_summary.json)",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError as exc:
        raise AuditError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be a JSON object")
    return value


def require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditError(f"{label} must be an integer")
    return value


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise AuditError(f"{label} must be finite")
    return number


def require_progress(
    condition_root: Path,
    task: str,
) -> tuple[list[dict[str, Any]], int]:
    path = condition_root / "progress" / f"{task}.json"
    progress = require_mapping(load_json(path), str(path))
    if progress.get("complete") is not True:
        raise AuditError(f"{path}: progress is not complete")
    episodes = progress.get("episode_results")
    if not isinstance(episodes, list):
        raise AuditError(f"{path}: episode_results must be a list")
    if len(episodes) != EXPECTED_EPISODES_PER_TASK:
        raise AuditError(
            f"{path}: expected {EXPECTED_EPISODES_PER_TASK} episodes, got {len(episodes)}"
        )

    parsed: list[dict[str, Any]] = []
    seeds: list[int] = []
    successes = 0
    for index, item in enumerate(episodes):
        row = require_mapping(item, f"{path}: episode_results[{index}]")
        seed = require_int(row.get("seed"), f"{path}: episode_results[{index}].seed")
        instruction = row.get("instruction")
        if not isinstance(instruction, str) or not instruction:
            raise AuditError(
                f"{path}: episode_results[{index}].instruction must be non-empty text"
            )
        success = row.get("success")
        if not isinstance(success, bool):
            raise AuditError(
                f"{path}: episode_results[{index}].success must be boolean"
            )
        seeds.append(seed)
        successes += int(success)
        parsed.append({"seed": seed, "instruction": instruction, "success": success})

    if len(set(seeds)) != EXPECTED_EPISODES_PER_TASK:
        raise AuditError(f"{path}: episode seeds are not unique: {seeds}")
    return parsed, successes


def audit_condition(
    split_root: Path,
    condition: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    condition_root = split_root / condition
    report_path = condition_root / "report.json"
    report = require_mapping(load_json(report_path), str(report_path))
    if report.get("schema") != EXPECTED_SCHEMA:
        raise AuditError(
            f"{report_path}: expected schema {EXPECTED_SCHEMA!r}, "
            f"got {report.get('schema')!r}"
        )
    if report.get("condition") != condition:
        raise AuditError(
            f"{report_path}: condition must be {condition!r}, "
            f"got {report.get('condition')!r}"
        )
    if require_int(report.get("task_count"), f"{report_path}.task_count") != EXPECTED_TASK_COUNT:
        raise AuditError(f"{report_path}: task_count must be {EXPECTED_TASK_COUNT}")
    if require_int(report.get("episodes_per_task"), f"{report_path}.episodes_per_task") != EXPECTED_EPISODES_PER_TASK:
        raise AuditError(
            f"{report_path}: episodes_per_task must be {EXPECTED_EPISODES_PER_TASK}"
        )
    if require_int(report.get("total_episodes"), f"{report_path}.total_episodes") != EXPECTED_CELL_EPISODES:
        raise AuditError(
            f"{report_path}: total_episodes must be {EXPECTED_CELL_EPISODES}"
        )

    report_rows = report.get("tasks")
    if not isinstance(report_rows, list) or len(report_rows) != EXPECTED_TASK_COUNT:
        raise AuditError(f"{report_path}: tasks must contain exactly {EXPECTED_TASK_COUNT} rows")
    rows_by_task: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(report_rows):
        row = require_mapping(raw_row, f"{report_path}.tasks[{index}]")
        task = row.get("task")
        if not isinstance(task, str) or not task:
            raise AuditError(f"{report_path}.tasks[{index}].task must be non-empty text")
        if task in rows_by_task:
            raise AuditError(f"{report_path}: duplicate task {task!r}")
        rows_by_task[task] = row

    expected_tasks = set(rows_by_task)
    episodes_by_task: dict[str, list[dict[str, Any]]] = {}
    successes_total = 0
    task_details: list[dict[str, Any]] = []
    for task in sorted(expected_tasks):
        episodes, successes = require_progress(condition_root, task)
        row = rows_by_task[task]
        if require_int(row.get("episodes"), f"{report_path}.tasks[{task}].episodes") != EXPECTED_EPISODES_PER_TASK:
            raise AuditError(f"{report_path}: task {task!r} has incorrect episode count")
        reported_successes = require_int(
            row.get("successes"), f"{report_path}.tasks[{task}].successes"
        )
        if reported_successes != successes:
            raise AuditError(
                f"{report_path}: task {task!r} reports {reported_successes} successes, "
                f"progress contains {successes}"
            )
        reported_rate = require_number(
            row.get("success_rate"), f"{report_path}.tasks[{task}].success_rate"
        )
        if not math.isclose(
            reported_rate,
            successes / EXPECTED_EPISODES_PER_TASK,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AuditError(f"{report_path}: task {task!r} success_rate disagrees with progress")
        episodes_by_task[task] = episodes
        successes_total += successes
        task_details.append(
            {
                "task": task,
                "episodes": EXPECTED_EPISODES_PER_TASK,
                "successes": successes,
                "success_rate": successes / EXPECTED_EPISODES_PER_TASK,
            }
        )

    reported_total_successes = require_int(
        report.get("total_successes"), f"{report_path}.total_successes"
    )
    if reported_total_successes != successes_total:
        raise AuditError(
            f"{report_path}: total_successes {reported_total_successes} "
            f"does not match progress {successes_total}"
        )
    reported_micro = require_number(
        report.get("micro_success_rate"), f"{report_path}.micro_success_rate"
    )
    if not math.isclose(
        reported_micro,
        successes_total / EXPECTED_CELL_EPISODES,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AuditError(f"{report_path}: micro_success_rate disagrees with progress")
    reported_macro = require_number(
        report.get("macro_success_rate"), f"{report_path}.macro_success_rate"
    )
    expected_macro = successes_total / EXPECTED_CELL_EPISODES
    # Every task has exactly eight episodes, so macro and micro must agree.
    if not math.isclose(reported_macro, expected_macro, rel_tol=0.0, abs_tol=1e-12):
        raise AuditError(f"{report_path}: macro_success_rate disagrees with progress")

    return {
        "condition": condition,
        "report": str(report_path),
        "tasks": task_details,
        "total_episodes": EXPECTED_CELL_EPISODES,
        "total_successes": successes_total,
        "success_rate": successes_total / EXPECTED_CELL_EPISODES,
    }, episodes_by_task


def audit_split(root: Path, split: str) -> dict[str, Any]:
    split_root = root / split
    if not split_root.is_dir():
        raise AuditError(f"missing split directory: {split_root}")
    baseline, baseline_episodes = audit_condition(split_root, "baseline")
    zeva, zeva_episodes = audit_condition(split_root, "zeva")

    if set(baseline_episodes) != set(zeva_episodes):
        raise AuditError(f"{split}: Base/ZeVA task sets differ")
    task_rows: list[dict[str, Any]] = []
    for task in sorted(baseline_episodes):
        base_items = baseline_episodes[task]
        zeva_items = zeva_episodes[task]
        base_pairs = [(item["seed"], item["instruction"]) for item in base_items]
        zeva_pairs = [(item["seed"], item["instruction"]) for item in zeva_items]
        if base_pairs != zeva_pairs:
            raise AuditError(f"{split}/{task}: Base/ZeVA seed or instruction pairing differs")
        base_successes = sum(int(item["success"]) for item in base_items)
        zeva_successes = sum(int(item["success"]) for item in zeva_items)
        task_rows.append(
            {
                "task": task,
                "episodes": EXPECTED_EPISODES_PER_TASK,
                "baseline_successes": base_successes,
                "zeva_successes": zeva_successes,
                "delta_count": zeva_successes - base_successes,
                "delta_rate": (zeva_successes - base_successes) / EXPECTED_EPISODES_PER_TASK,
            }
        )

    delta_count = zeva["total_successes"] - baseline["total_successes"]
    return {
        "split": split,
        "baseline_report": baseline,
        "zeva_report": zeva,
        "paired_episodes": EXPECTED_CELL_EPISODES,
        "baseline_successes": baseline["total_successes"],
        "zeva_successes": zeva["total_successes"],
        "delta_count": delta_count,
        "delta_rate": delta_count / EXPECTED_CELL_EPISODES,
        "cell_pass": delta_count >= 0,
        "tasks": task_rows,
    }


def atomic_write_json(destination: Path, payload: dict[str, Any]) -> None:
    destination = destination.resolve()
    if not destination.parent.is_dir():
        raise AuditError(f"output directory does not exist: {destination.parent}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".partial",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def failure_payload(root: Path, errors: list[str]) -> dict[str, Any]:
    return {
        "schema": "zeva-robotwin-v11-initial-validation-v1",
        "passed": False,
        "accepted_for_four_cell_validation": False,
        "root": str(root),
        "criteria": {
            "expected_splits": list(EXPECTED_SPLITS),
            "expected_task_count": EXPECTED_TASK_COUNT,
            "expected_episodes_per_task": EXPECTED_EPISODES_PER_TASK,
            "expected_episodes_per_cell": EXPECTED_CELL_EPISODES,
            "each_cell_delta_at_least": 0,
            "minimum_combined_delta_count": MINIMUM_TOTAL_GAIN,
            "minimum_combined_delta_denominator": 2 * EXPECTED_CELL_EPISODES,
            "task_aggregate_is_diagnostic_only": True,
        },
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    destination = (args.output or root / "validation_summary.json").resolve()
    errors: list[str] = []
    splits: list[dict[str, Any]] = []
    if not root.is_dir():
        errors.append(f"missing validation root: {root}")
    else:
        for split in EXPECTED_SPLITS:
            try:
                splits.append(audit_split(root, split))
            except AuditError as exc:
                errors.append(str(exc))

    if errors:
        payload = failure_payload(root, errors)
    else:
        total_delta = sum(int(item["delta_count"]) for item in splits)
        negative_cells = [item["split"] for item in splits if item["delta_count"] < 0]
        task_aggregate: list[dict[str, Any]] = []
        all_tasks = sorted({row["task"] for item in splits for row in item["tasks"]})
        for task in all_tasks:
            rows = [row for item in splits for row in item["tasks"] if row["task"] == task]
            baseline_successes = sum(int(row["baseline_successes"]) for row in rows)
            zeva_successes = sum(int(row["zeva_successes"]) for row in rows)
            task_aggregate.append(
                {
                    "task": task,
                    "episodes": len(rows) * EXPECTED_EPISODES_PER_TASK,
                    "baseline_successes": baseline_successes,
                    "zeva_successes": zeva_successes,
                    "delta_count": zeva_successes - baseline_successes,
                    "delta_rate": (zeva_successes - baseline_successes)
                    / (len(rows) * EXPECTED_EPISODES_PER_TASK),
                }
            )
        accepted = not negative_cells and total_delta >= MINIMUM_TOTAL_GAIN
        payload = {
            "schema": "zeva-robotwin-v11-initial-validation-v1",
            "passed": accepted,
            "accepted_for_four_cell_validation": accepted,
            "root": str(root),
            "criteria": {
                "expected_splits": list(EXPECTED_SPLITS),
                "expected_task_count": EXPECTED_TASK_COUNT,
                "expected_episodes_per_task": EXPECTED_EPISODES_PER_TASK,
                "expected_episodes_per_cell": EXPECTED_CELL_EPISODES,
                "each_cell_delta_at_least": 0,
                "minimum_combined_delta_count": MINIMUM_TOTAL_GAIN,
                "minimum_combined_delta_denominator": 2 * EXPECTED_CELL_EPISODES,
                "task_aggregate_is_diagnostic_only": True,
            },
            "splits": splits,
            "total_paired_episodes": 2 * EXPECTED_CELL_EPISODES,
            "baseline_successes": sum(int(item["baseline_successes"]) for item in splits),
            "zeva_successes": sum(int(item["zeva_successes"]) for item in splits),
            "total_delta_count": total_delta,
            "total_delta_rate": total_delta / (2 * EXPECTED_CELL_EPISODES),
            "negative_cells": negative_cells,
            "task_aggregate": task_aggregate,
            "errors": [],
        }

    try:
        atomic_write_json(destination, payload)
    except AuditError as exc:
        print(f"audit failed before summary write: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
