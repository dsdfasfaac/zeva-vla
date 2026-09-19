#!/usr/bin/env python3
"""Audit and compare frozen Base, parent, and PIM multi-attempt development runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_condition(root: Path, expected_reset: str) -> tuple[list[dict], dict]:
    report = json.loads((root / "report.json").read_text())
    if report.get("total_episodes") != 80:
        raise RuntimeError(f"{root}: expected 80 episodes")
    rows: list[dict] = []
    for progress in sorted((root / "progress").glob("*.json")):
        payload = json.loads(progress.read_text())
        if not payload.get("complete") or len(payload.get("episode_results", [])) != 8:
            raise RuntimeError(f"{progress}: incomplete")
        if payload["identity"].get("max_policy_attempts") != 4:
            raise RuntimeError(f"{progress}: wrong attempt budget")
        if payload["identity"].get("attempt_reset_scope") != expected_reset:
            raise RuntimeError(f"{progress}: wrong reset scope")
        task = progress.stem
        for episode in payload["episode_results"]:
            attempts = episode.get("attempts", [])
            if not 1 <= len(attempts) <= 4:
                raise RuntimeError(f"{progress}: invalid attempt count")
            if any(bool(a["success"]) for a in attempts[:-1]):
                raise RuntimeError(f"{progress}: continued after success")
            if bool(episode["success"]) != any(bool(a["success"]) for a in attempts):
                raise RuntimeError(f"{progress}: cumulative outcome mismatch")
            expected_scopes = ["episode"] + [expected_reset] * (len(attempts) - 1)
            if [a.get("reset_scope") for a in attempts] != expected_scopes:
                raise RuntimeError(f"{progress}: per-attempt reset trace mismatch")
            rows.append({"task": task, **episode})
    if len(rows) != 80:
        raise RuntimeError(f"{root}: only {len(rows)} episode traces")
    videos = list((root / "results").glob("**/*success-*.mp4"))
    expected_videos = sum(len(row["attempts"]) for row in rows)
    if len(videos) != expected_videos or report.get("total_attempts_executed") != expected_videos:
        raise RuntimeError(f"{root}: attempt video count mismatch")
    return rows, report


def curve(rows: list[dict]) -> dict:
    exact, cumulative, attempted = [], [], []
    for index in range(4):
        attempted.append(sum(len(row["attempts"]) > index for row in rows))
        exact.append(sum(len(row["attempts"]) > index and bool(row["attempts"][index]["success"])
                         for row in rows))
        cumulative.append(sum(any(bool(a["success"]) for a in row["attempts"][: index + 1])
                              for row in rows))
    return {"attempted": attempted, "successes_at_attempt": exact,
            "cumulative_successes": cumulative,
            "cumulative_success_rates": [value / len(rows) for value in cumulative]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--pim-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = [root / "seed_manifest.json" for root in
                 (args.base_root, args.parent_root, args.pim_root)]
    manifest_shas = [sha256(path) for path in manifests]
    if len(set(manifest_shas)) != 1:
        raise RuntimeError("condition seed manifests differ")
    base, base_report = load_condition(args.base_root / "baseline", "episode")
    parent, parent_report = load_condition(args.parent_root / "zeva", "episode")
    pim, pim_report = load_condition(args.pim_root / "zeva", "attempt")
    by_key = lambda rows: {(row["task"], int(row["episode_index"])): row for row in rows}
    base_map, parent_map, pim_map = map(by_key, (base, parent, pim))
    if not (base_map.keys() == parent_map.keys() == pim_map.keys()):
        raise RuntimeError("episode identities differ")
    for key in base_map:
        identities = {(row["seed"], row["instruction"])
                      for row in (base_map[key], parent_map[key], pim_map[key])}
        if len(identities) != 1:
            raise RuntimeError(f"seed/instruction mismatch for {key}")

    base_curve, parent_curve, pim_curve = map(curve, (base, parent, pim))
    checks = {
        "complete_80_episode_coverage": True,
        "identical_seed_instruction_manifest": True,
        "all_attempt_videos_present": True,
        "first_attempt_pim_not_worse_than_parent":
            pim_curve["cumulative_successes"][0] >= parent_curve["cumulative_successes"][0],
        "pim_has_at_least_four_extra_cumulative_successes_at_attempt4":
            pim_curve["cumulative_successes"][3] - parent_curve["cumulative_successes"][3] >= 4,
    }
    payload = {
        "schema": "zeva-pim-multiattempt-development-report-v1",
        "seed_manifest_sha256": manifest_shas[0],
        "formal_success_labels_used": False,
        "episodes": 80,
        "max_attempts": 4,
        "conditions": {
            "base": {"reset_scope": "episode", "curve": base_curve,
                     "cumulative_successes": base_report["total_successes"]},
            "parent": {"reset_scope": "episode", "curve": parent_curve,
                       "cumulative_successes": parent_report["total_successes"]},
            "pim": {"reset_scope": "attempt", "curve": pim_curve,
                    "cumulative_successes": pim_report["total_successes"]},
        },
        "pim_minus_parent": {
            "first_attempt_successes": pim_curve["cumulative_successes"][0] - parent_curve["cumulative_successes"][0],
            "attempt4_cumulative_successes": pim_curve["cumulative_successes"][3] - parent_curve["cumulative_successes"][3],
            "attempt4_percentage_points": 100 * (pim_curve["cumulative_successes"][3]
                                                   - parent_curve["cumulative_successes"][3]) / 80,
        },
        "gate": {"passed": all(checks.values()), "checks": checks,
                 "fixed_before_results": True,
                 "required_extra_cumulative_successes": 4},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
