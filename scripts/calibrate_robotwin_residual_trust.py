#!/usr/bin/env python3
"""Create a task-language residual trust table from disjoint closed-loop validation.

The selector never reads formal-test outcomes.  For each task it chooses the
smallest non-zero residual scale that achieves the best validation success
count, but only when that count strictly exceeds the paired PI0.5 baseline.
Otherwise the task receives scale 0 and is exactly routed through PI0.5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_progress(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        row = _load_json(path)
        task = str(row.get("identity", {}).get("task_name", path.stem))
        if not row.get("complete"):
            raise RuntimeError(f"Calibration progress is incomplete: {path}")
        episodes = row.get("episode_results", [])
        if len(episodes) != int(row.get("completed_episodes", -1)):
            raise RuntimeError(f"Episode/result count mismatch: {path}")
        if sum(bool(item["success"]) for item in episodes) != int(row.get("successes", -1)):
            raise RuntimeError(f"Success count mismatch: {path}")
        rows[task] = row
    if not rows:
        raise RuntimeError(f"No progress files found under {root}")
    return rows


def _episode_contract(row: dict[str, Any]) -> list[tuple[int, str]]:
    return [(int(item["seed"]), str(item["instruction"])) for item in row["episode_results"]]


def _manifest_seed_sets(path: Path) -> dict[str, set[int]]:
    payload = _load_json(path)
    result: dict[str, set[int]] = {}
    for task, row in payload.get("tasks", {}).items():
        if isinstance(row, list):
            seeds = [item["seed"] if isinstance(item, dict) else item for item in row]
        else:
            seeds = row.get("seeds", row.get("fixed_seed_values", []))
        result[str(task)] = {int(seed) for seed in seeds}
    if not result:
        raise RuntimeError(f"Seed manifest has no tasks: {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-progress", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="SCALE=PROGRESS_DIR",
        help="Closed-loop ZeVA validation progress for one scale; repeat as needed.",
    )
    parser.add_argument("--validation-seed-manifest", type=Path, required=True)
    parser.add_argument("--forbidden-seed-manifest", type=Path, required=True)
    parser.add_argument("--minimum-success-gain", type=int, default=1)
    args = parser.parse_args()

    candidates: dict[float, Path] = {}
    for item in args.candidate:
        scale_text, separator, path_text = item.partition("=")
        if not separator:
            raise ValueError(f"Candidate must be SCALE=PROGRESS_DIR, got {item!r}")
        scale = float(scale_text)
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError(f"Candidate scale must be in (0, 1], got {scale_text!r}")
        if scale in candidates:
            raise ValueError(f"Duplicate candidate scale: {scale}")
        candidates[scale] = Path(path_text).resolve()

    validation_seeds = _manifest_seed_sets(args.validation_seed_manifest.resolve())
    forbidden_seeds = _manifest_seed_sets(args.forbidden_seed_manifest.resolve())
    for task in sorted(set(validation_seeds).intersection(forbidden_seeds)):
        overlap = sorted(validation_seeds[task].intersection(forbidden_seeds[task]))
        if overlap:
            raise RuntimeError(f"Validation/formal-test seed leakage for {task}: {overlap}")

    baseline = _load_progress(args.baseline_progress.resolve())
    candidate_rows = {scale: _load_progress(path) for scale, path in candidates.items()}
    task_names = sorted(baseline)
    if set(task_names) != set(validation_seeds):
        raise RuntimeError("Baseline tasks differ from the validation seed manifest.")
    for scale, rows in candidate_rows.items():
        if set(rows) != set(task_names):
            raise RuntimeError(f"Candidate {scale} task set differs from baseline.")
        for task in task_names:
            if _episode_contract(rows[task]) != _episode_contract(baseline[task]):
                raise RuntimeError(f"Candidate {scale} is not exactly paired for {task}.")

    selected: dict[str, float] = {}
    task_evidence: dict[str, Any] = {}
    for task in task_names:
        baseline_successes = int(baseline[task]["successes"])
        ranked = sorted(
            (
                int(rows[task]["successes"]),
                -scale,  # on ties choose the smallest residual scale
                scale,
            )
            for scale, rows in candidate_rows.items()
        )
        best_successes, _, best_scale = max(ranked)
        gain = best_successes - baseline_successes
        chosen_scale = best_scale if gain >= args.minimum_success_gain else 0.0
        selected[task] = chosen_scale
        task_evidence[task] = {
            "episodes": int(baseline[task]["completed_episodes"]),
            "baseline_successes": baseline_successes,
            "candidate_successes": {
                str(scale): int(candidate_rows[scale][task]["successes"])
                for scale in sorted(candidate_rows)
            },
            "best_candidate_scale": best_scale,
            "best_candidate_successes": best_successes,
            "validation_gain": gain,
            "selected_scale": chosen_scale,
        }
    if not any(scale > 0 for scale in selected.values()):
        raise RuntimeError("No task has a strictly positive closed-loop validation gain.")

    calibration = {
        "schema": "zeva-robotwin-closed-loop-residual-calibration-v1",
        "selection_rule": (
            "per task: maximize paired closed-loop validation successes; ties choose the "
            "smallest residual scale; use scale 0 unless gain >= minimum_success_gain"
        ),
        "minimum_success_gain": args.minimum_success_gain,
        "validation_seed_manifest": str(args.validation_seed_manifest.resolve()),
        "validation_seed_manifest_sha256": _sha256(args.validation_seed_manifest.resolve()),
        "forbidden_formal_seed_manifest": str(args.forbidden_seed_manifest.resolve()),
        "forbidden_formal_seed_manifest_sha256": _sha256(args.forbidden_seed_manifest.resolve()),
        "seed_sets_disjoint": True,
        "test_metrics_used": False,
        "baseline_progress": str(args.baseline_progress.resolve()),
        "candidate_progress": {str(scale): str(path) for scale, path in sorted(candidates.items())},
        "tasks": task_evidence,
    }

    checkpoint = torch.load(args.adapter.resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("schema") not in {
        "zeva-robotwin-stage2-adapter-v5",
        "zeva-robotwin-pi05-adapter-v4",
    }:
        raise RuntimeError(f"Unsupported adapter schema: {checkpoint.get('schema')!r}")
    checkpoint["deployment_task_residual_scales"] = selected
    checkpoint["deployment_default_residual_scale"] = 0.0
    checkpoint["deployment_residual_calibration"] = calibration

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    report_path = args.output.with_suffix(args.output.suffix + ".calibration.json")
    report_path.write_text(json.dumps(calibration, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "scales": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
