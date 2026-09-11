"""Check that two declared ZTE learning curves differ only in loss reduction.

This is an experiment-manifest audit, not a representation or policy gate.
Live world size, completed steps, and outcome quality require separate evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def differences(first, second, path=""):
    if isinstance(first, dict) and isinstance(second, dict):
        result = []
        for key in sorted(first.keys() | second.keys()):
            child = f"{path}.{key}" if path else key
            if key not in first or key not in second:
                result.append({"path": child, "missing_from": "first" if key not in first else "second"})
            else:
                result.extend(differences(first[key], second[key], child))
        return result
    if type(first) is not type(second) or first != second:
        return [{"path": path, "first": first, "second": second}]
    return []


def audit(first: dict, second: dict) -> dict:
    allowed = {"train_args.save_dir", "train_args.prediction_loss_reduction"}
    delta = differences(first, second)
    unexpected = [item for item in delta if item["path"] not in allowed or "missing_from" in item]
    reductions = [item.get("train_args", {}).get("prediction_loss_reduction") for item in (first, second)]
    expected_reductions = set(reductions) == {"mean_coordinate_huber", "vector_mse"}
    separate_output = first.get("train_args", {}).get("save_dir") != second.get("train_args", {}).get("save_dir")
    required_paths = (
        "source_files.trainer", "source_files.encoder", "statistics_sha256",
        "goal_embeddings_sha256", "zte_config.action_prediction_context",
        "contract.executed_horizon", "contract.policy_horizon", "grouping.task_names",
    )
    missing = []
    for index, item in enumerate((first, second)):
        for path in required_paths:
            value = item
            for key in path.split("."):
                value = value.get(key) if isinstance(value, dict) else None
            if value is None or value == "" or value == []:
                missing.append({"source": index, "path": path})
    settings = all(
        item.get("train_args", {}).get("steps") == 4096
        and item.get("train_args", {}).get("batch_size") == 8
        and item.get("train_args", {}).get("action_prediction_context") == "phase"
        and item.get("train_args", {}).get("task_paired_batches") is True
        and item.get("schema") == "zeva-robotwin-zte-stage1-v2"
        and item.get("zte_config", {}).get("action_prediction_context") == "phase"
        and item.get("contract", {}).get("executed_horizon") == 15
        and item.get("contract", {}).get("policy_horizon") == 50
        for item in (first, second)
    )
    return {
        "schema": "zeva-zte-loss-control-manifest-audit-v1",
        "passed": not unexpected and not missing and expected_reductions and separate_output and settings,
        "status": "protocol_only",
        "stage1_gate_passed": False,
        "expected_reductions": expected_reductions,
        "separate_output_directories": separate_output,
        "preregistered_settings": settings,
        "allowed_difference_paths": sorted(allowed),
        "actual_differences": delta,
        "unexpected_differences": unexpected,
        "missing_required_fields": missing,
        "limits": [
            "Manifest equality is not proof of convergence or successful closed-loop control.",
            "Check live processes or saved checkpoints for actual four-rank execution and completed steps.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.first.resolve() == args.second.resolve():
        parser.error("Two distinct experiment manifests are required")
    if args.output.exists() or args.output.resolve() in (args.first.resolve(), args.second.resolve()):
        parser.error("Use a fresh audit output path; never overwrite source manifests")
    blobs = [path.read_bytes() for path in (args.first, args.second)]
    report = audit(*(json.loads(blob) for blob in blobs))
    report["sources"] = [
        {"path": str(path.resolve()), "sha256": hashlib.sha256(blob).hexdigest()}
        for path, blob in zip((args.first, args.second), blobs, strict=True)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
