#!/usr/bin/env python3
"""Calibrate task-language residual trust across disjoint closed-loop splits.

Unlike the single-split calibrator, a residual scale is eligible only when it
does not regress on *any* validation split and clears a preregistered aggregate
success-gain threshold.  Formal-test results are never an input.
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


def _named_path(value: str, *, option: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"{option} must be NAME=PATH, got {value!r}")
    return name, Path(path).resolve()


def _candidate(value: str) -> tuple[str, float, Path]:
    split_scale, separator, path = value.partition("=")
    split, colon, scale_text = split_scale.partition(":")
    if not separator or not colon or not split or not scale_text or not path:
        raise ValueError(f"--candidate must be SPLIT:SCALE=PATH, got {value!r}")
    scale = float(scale_text)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError(f"Candidate scale must be in (0, 1], got {scale_text!r}")
    return split, scale, Path(path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True, metavar="NAME=BASELINE_PROGRESS")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="SPLIT:SCALE=PROGRESS_DIR",
    )
    parser.add_argument(
        "--validation-seed-manifest",
        action="append",
        required=True,
        metavar="NAME=MANIFEST",
    )
    parser.add_argument("--forbidden-seed-manifest", type=Path, required=True)
    parser.add_argument("--minimum-total-success-gain", type=int, default=3)
    args = parser.parse_args()

    split_paths = dict(_named_path(value, option="--split") for value in args.split)
    manifest_paths = dict(
        _named_path(value, option="--validation-seed-manifest")
        for value in args.validation_seed_manifest
    )
    if len(split_paths) < 2:
        raise RuntimeError("Multi-split calibration requires at least two validation splits.")
    if set(split_paths) != set(manifest_paths):
        raise RuntimeError("Validation split and seed-manifest names differ.")

    candidate_paths: dict[str, dict[float, Path]] = {name: {} for name in split_paths}
    for value in args.candidate:
        split, scale, path = _candidate(value)
        if split not in candidate_paths:
            raise ValueError(f"Candidate references unknown split {split!r}")
        if scale in candidate_paths[split]:
            raise ValueError(f"Duplicate candidate {split}:{scale}")
        candidate_paths[split][scale] = path
    scale_sets = {frozenset(rows) for rows in candidate_paths.values()}
    if len(scale_sets) != 1 or not next(iter(scale_sets), frozenset()):
        raise RuntimeError("Every validation split must contain the same non-empty scale set.")
    scales = sorted(next(iter(scale_sets)))

    seed_sets = {name: _manifest_seed_sets(path) for name, path in manifest_paths.items()}
    forbidden_path = args.forbidden_seed_manifest.resolve()
    all_seed_sets = {**seed_sets, "formal-test": _manifest_seed_sets(forbidden_path)}
    names = sorted(all_seed_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for task in sorted(set(all_seed_sets[left]).intersection(all_seed_sets[right])):
                overlap = sorted(all_seed_sets[left][task].intersection(all_seed_sets[right][task]))
                if overlap:
                    raise RuntimeError(
                        f"Seed leakage between {left} and {right} for {task}: {overlap}"
                    )

    baselines = {name: _load_progress(path) for name, path in split_paths.items()}
    candidates = {
        name: {scale: _load_progress(path) for scale, path in paths.items()}
        for name, paths in candidate_paths.items()
    }
    task_names = sorted(next(iter(baselines.values())))
    for split, baseline in baselines.items():
        if set(baseline) != set(task_names) or set(baseline) != set(seed_sets[split]):
            raise RuntimeError(f"Task set mismatch in validation split {split!r}.")
        for scale, rows in candidates[split].items():
            if set(rows) != set(task_names):
                raise RuntimeError(f"Candidate {split}:{scale} task set differs from baseline.")
            for task in task_names:
                if _episode_contract(rows[task]) != _episode_contract(baseline[task]):
                    raise RuntimeError(f"Candidate {split}:{scale} is not exactly paired for {task}.")

    selected: dict[str, float] = {}
    task_evidence: dict[str, Any] = {}
    for task in task_names:
        scale_evidence: dict[str, Any] = {}
        eligible: list[tuple[int, float]] = []
        for scale in scales:
            split_gains = {
                split: int(candidates[split][scale][task]["successes"])
                - int(baselines[split][task]["successes"])
                for split in sorted(split_paths)
            }
            total_gain = sum(split_gains.values())
            is_eligible = (
                all(gain >= 0 for gain in split_gains.values())
                and total_gain >= args.minimum_total_success_gain
            )
            scale_evidence[str(scale)] = {
                "split_success_gains": split_gains,
                "total_success_gain": total_gain,
                "eligible": is_eligible,
            }
            if is_eligible:
                eligible.append((total_gain, scale))
        # Maximize aggregate validation gain; on ties select the smaller scale.
        chosen = max(eligible, key=lambda item: (item[0], -item[1]))[1] if eligible else 0.0
        selected[task] = chosen
        task_evidence[task] = {
            "split_baseline_successes": {
                split: int(baselines[split][task]["successes"])
                for split in sorted(split_paths)
            },
            "scales": scale_evidence,
            "selected_scale": chosen,
        }
    if not any(scale > 0 for scale in selected.values()):
        raise RuntimeError("No task passes the multi-split residual trust gate.")

    calibration = {
        "schema": "zeva-robotwin-multisplit-residual-calibration-v1",
        "selection_rule": (
            "per task and scale: require non-negative paired success gain on every "
            "validation split and aggregate gain >= minimum_total_success_gain; "
            "maximize aggregate gain and break ties toward the smaller scale; otherwise scale 0"
        ),
        "minimum_total_success_gain": args.minimum_total_success_gain,
        "validation_splits": {
            name: {
                "baseline_progress": str(split_paths[name]),
                "candidate_progress": {
                    str(scale): str(candidate_paths[name][scale]) for scale in scales
                },
                "seed_manifest": str(manifest_paths[name]),
                "seed_manifest_sha256": _sha256(manifest_paths[name]),
            }
            for name in sorted(split_paths)
        },
        "forbidden_formal_seed_manifest": str(forbidden_path),
        "forbidden_formal_seed_manifest_sha256": _sha256(forbidden_path),
        "seed_sets_pairwise_disjoint": True,
        "test_metrics_used": False,
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
