#!/usr/bin/env python3
"""Join paired development-split RoboTwin outcome traces into a dataset.

Example::

    python scripts/robotwin_eval/join_validate_outcome_traces.py \
      --baseline-dir /path/to/paired/baseline/traces \
      --zeva-dir /path/to/paired/zeva/traces \
      --output /path/to/paired/outcome_dataset.json

The join key is ``(split, protocol, task, seed, instruction)``.  A duplicate,
missing pair, provenance mismatch, instruction mismatch, or missing success
label aborts without writing a dataset.  This utility is deliberately limited
to development/validation splits: an episode success is an outcome label for
the complete rollout, not a per-decision failure-causality label, and final
test labels must never be used to supervise an outcome selector.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from outcome_trace import TRACE_SCHEMA
    from outcome_trace import OutcomeTraceError
    from outcome_trace import OUTCOME_LABEL_SCOPE
    from outcome_trace import SUPERVISION_SPLITS
    from outcome_trace import load_episode_trace
except ImportError:  # pragma: no cover - package import path in unit tests.
    from .outcome_trace import TRACE_SCHEMA
    from .outcome_trace import OutcomeTraceError
    from .outcome_trace import OUTCOME_LABEL_SCOPE
    from .outcome_trace import SUPERVISION_SPLITS
    from .outcome_trace import load_episode_trace


def _discover(
    root: Path, expected_condition: str
) -> dict[tuple[str, str, str, int, str], tuple[dict[str, Any], Path]]:
    if not root.is_dir():
        raise OutcomeTraceError(f"trace directory does not exist: {root}")
    found: dict[tuple[str, str, str, int, str], tuple[dict[str, Any], Path]] = {}
    candidates = sorted(path for path in root.rglob("*.json") if "__episode" in path.name)
    if not candidates:
        raise OutcomeTraceError(f"no episode trace files found below {root}")
    for path in candidates:
        try:
            payload = load_episode_trace(path)
        except OutcomeTraceError as error:
            raise OutcomeTraceError(f"invalid trace file {path}: {error}") from error
        if payload.get("schema") != TRACE_SCHEMA:
            raise OutcomeTraceError(f"{path}: unexpected trace schema {payload.get('schema')!r}")
        if payload.get("condition") != expected_condition:
            raise OutcomeTraceError(
                f"{path}: expected condition {expected_condition!r}, "
                f"got {payload.get('condition')!r}"
            )
        if not isinstance(payload.get("success"), bool):
            raise OutcomeTraceError(f"{path}: missing boolean success label")
        key = (
            str(payload["split"]),
            str(payload["protocol"]),
            str(payload["task"]),
            int(payload["seed"]),
            str(payload["instruction"]),
        )
        if key in found:
            previous = found[key][1]
            raise OutcomeTraceError(
                "duplicate condition trace for split/protocol/task/seed/instruction: "
                f"{key!r} in {previous} and {path}"
            )
        found[key] = (payload, path)
    return found


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_dataset(
    baseline_dir: str | Path,
    zeva_dir: str | Path,
    *,
    output: str | Path,
    baseline_condition: str = "baseline",
    zeva_condition: str = "zeva",
) -> dict[str, Any]:
    baseline = _discover(Path(baseline_dir).expanduser(), baseline_condition)
    zeva = _discover(Path(zeva_dir).expanduser(), zeva_condition)
    provenance = {
        (payload["split"], payload["protocol"])
        for payload, _ in (*baseline.values(), *zeva.values())
    }
    if len(provenance) != 1:
        raise OutcomeTraceError(
            "Base/ZeVA traces must share exactly one split/protocol pair; "
            f"found {sorted(provenance)!r}"
        )
    split, protocol = next(iter(provenance))
    if split not in SUPERVISION_SPLITS:
        raise OutcomeTraceError(
            f"refusing to build supervision dataset from split {split!r}; "
            "success labels are development/validation-only and final-test traces "
            "are evaluation-only"
        )
    baseline_keys = set(baseline)
    zeva_keys = set(zeva)
    missing_zeva = sorted(baseline_keys - zeva_keys)
    extra_zeva = sorted(zeva_keys - baseline_keys)
    if missing_zeva or extra_zeva:
        raise OutcomeTraceError(
            "Base/ZeVA trace identity sets differ; "
            f"missing_zeva={missing_zeva[:5]}, extra_zeva={extra_zeva[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for key in sorted(baseline_keys):
        base_payload, base_path = baseline[key]
        zeva_payload, zeva_path = zeva[key]
        identity_fields = (
            "split",
            "protocol",
            "task",
            "seed",
            "episode_index",
            "instruction",
        )
        for field in identity_fields:
            if base_payload[field] != zeva_payload[field]:
                raise OutcomeTraceError(
                    f"paired identity mismatch for {key!r}: field {field!r} "
                    f"Base={base_payload[field]!r}, ZeVA={zeva_payload[field]!r}"
                )
        if not isinstance(base_payload.get("success"), bool) or not isinstance(
            zeva_payload.get("success"), bool
        ):
            raise OutcomeTraceError(f"paired trace {key!r} lacks complete success labels")
        rows.append(
            {
                "split": split,
                "protocol": protocol,
                "task": base_payload["task"],
                "seed": base_payload["seed"],
                "episode_index": base_payload["episode_index"],
                "instruction": base_payload["instruction"],
                "baseline_success": base_payload["success"],
                "zeva_success": zeva_payload["success"],
                "baseline_trace_file": str(base_path),
                "zeva_trace_file": str(zeva_path),
                "baseline": base_payload,
                "zeva": zeva_payload,
            }
        )

    total = len(rows)
    baseline_successes = sum(int(row["baseline_success"]) for row in rows)
    zeva_successes = sum(int(row["zeva_success"]) for row in rows)
    dataset = {
        "schema": "zeva-robotwin-outcome-dataset-v2",
        "trace_schema": TRACE_SCHEMA,
        "join_key": ["split", "protocol", "task", "seed", "instruction"],
        "split": split,
        "protocol": protocol,
        "outcome_label_scope": OUTCOME_LABEL_SCOPE,
        "success_labels_supervision": "development_only",
        "total_episodes": total,
        "baseline_successes": baseline_successes,
        "zeva_successes": zeva_successes,
        "baseline_success_rate": None if total == 0 else baseline_successes / total,
        "zeva_success_rate": None if total == 0 else zeva_successes / total,
        "episodes": rows,
    }
    _atomic_write(Path(output).expanduser(), dataset)
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--zeva-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-condition", default="baseline")
    parser.add_argument("--zeva-condition", default="zeva")
    args = parser.parse_args()
    try:
        dataset = build_dataset(
            args.baseline_dir,
            args.zeva_dir,
            output=args.output,
            baseline_condition=args.baseline_condition,
            zeva_condition=args.zeva_condition,
        )
    except (OSError, OutcomeTraceError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        f"Joined {dataset['total_episodes']} paired episodes: "
        f"Base {dataset['baseline_successes']}/{dataset['total_episodes']}, "
        f"ZeVA {dataset['zeva_successes']}/{dataset['total_episodes']} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
