#!/usr/bin/env python3
"""Independently audit the user-authorized three-way PIM formal run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = (
    "1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_condition(root: Path, expected_reset: str) -> tuple[list[dict], dict, int]:
    condition_root = root / "baseline"
    report = json.loads((condition_root / "report.json").read_text())
    if report.get("total_episodes") != 200 or report.get("task_count") != 10:
        raise RuntimeError(f"{root}: expected complete 10x20 report")

    authorization = json.loads((root / "formal_authorization.json").read_text())
    if not authorization.get("authorized_after_development_gate_failure"):
        raise RuntimeError(f"{root}: missing post-gate-failure authorization disclosure")

    rows: list[dict] = []
    for progress in sorted((condition_root / "progress").glob("*.json")):
        payload = json.loads(progress.read_text())
        episodes = payload.get("episode_results", [])
        if not payload.get("complete") or len(episodes) != 20:
            raise RuntimeError(f"{progress}: incomplete")
        identity = payload.get("identity", {})
        if identity.get("max_policy_attempts") != 4:
            raise RuntimeError(f"{progress}: wrong attempt budget")
        if identity.get("attempt_reset_scope") != expected_reset:
            raise RuntimeError(f"{progress}: wrong reset scope")
        task = progress.stem
        for episode in episodes:
            attempts = episode.get("attempts", [])
            if not 1 <= len(attempts) <= 4:
                raise RuntimeError(f"{progress}: invalid attempt count")
            if any(bool(attempt["success"]) for attempt in attempts[:-1]):
                raise RuntimeError(f"{progress}: continued after success")
            if bool(episode["success"]) != any(bool(a["success"]) for a in attempts):
                raise RuntimeError(f"{progress}: cumulative outcome mismatch")
            expected_scopes = ["episode"] + [expected_reset] * (len(attempts) - 1)
            if [a.get("reset_scope") for a in attempts] != expected_scopes:
                raise RuntimeError(f"{progress}: per-attempt reset trace mismatch")
            rows.append({"task": task, **episode})
    if len(rows) != 200:
        raise RuntimeError(f"{root}: only {len(rows)} episode traces")

    videos = list((condition_root / "results").glob("**/*success-*.mp4"))
    expected_videos = sum(len(row["attempts"]) for row in rows)
    if len(videos) != expected_videos:
        raise RuntimeError(f"{root}: {len(videos)} videos != {expected_videos} attempts")
    if report.get("total_attempts_executed") != expected_videos:
        raise RuntimeError(f"{root}: report attempt count mismatch")
    if report.get("total_successes") != sum(bool(row["success"]) for row in rows):
        raise RuntimeError(f"{root}: report success count mismatch")
    return rows, report, len(videos)


def curve(rows: list[dict]) -> dict:
    attempted, exact, cumulative = [], [], []
    for index in range(4):
        attempted.append(sum(len(row["attempts"]) > index for row in rows))
        exact.append(sum(
            len(row["attempts"]) > index and bool(row["attempts"][index]["success"])
            for row in rows
        ))
        cumulative.append(sum(
            any(bool(a["success"]) for a in row["attempts"][: index + 1])
            for row in rows
        ))
    return {
        "attempted": attempted,
        "successes_at_attempt": exact,
        "independent_success_rates": [
            successes / count if count else None
            for successes, count in zip(exact, attempted)
        ],
        "exact_success_contribution_over_all_episodes": [
            successes / len(rows) for successes in exact
        ],
        "cumulative_successes": cumulative,
        "cumulative_success_rates": [value / len(rows) for value in cumulative],
    }


def per_task(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    return {task: curve(task_rows) for task, task_rows in sorted(grouped.items())}


def paired_delta(left: list[dict], right: list[dict]) -> dict:
    left_map = {(row["task"], int(row["episode_index"])): row for row in left}
    right_map = {(row["task"], int(row["episode_index"])): row for row in right}
    left_only = right_only = ties_success = ties_failure = 0
    for key in sorted(left_map):
        left_success = bool(left_map[key]["success"])
        right_success = bool(right_map[key]["success"])
        if left_success and not right_success:
            left_only += 1
        elif right_success and not left_success:
            right_only += 1
        elif left_success:
            ties_success += 1
        else:
            ties_failure += 1
    return {
        "left_only_successes": left_only,
        "right_only_successes": right_only,
        "net_successes": left_only - right_only,
        "percentage_points": 100 * (left_only - right_only) / len(left),
        "both_success": ties_success,
        "both_failure": ties_failure,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--pim-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = (args.base_root, args.parent_root, args.pim_root)
    manifest_shas = [sha256(root / "seed_manifest.json") for root in roots]
    if len(set(manifest_shas)) != 1 or manifest_shas[0] != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"unexpected condition manifest SHAs: {manifest_shas}")

    base, base_report, base_videos = load_condition(args.base_root, "episode")
    parent, parent_report, parent_videos = load_condition(args.parent_root, "episode")
    pim, pim_report, pim_videos = load_condition(args.pim_root, "attempt")
    maps = [
        {(row["task"], int(row["episode_index"])): row for row in rows}
        for rows in (base, parent, pim)
    ]
    if not (maps[0].keys() == maps[1].keys() == maps[2].keys()):
        raise RuntimeError("episode identities differ")
    for key in maps[0]:
        identities = {(mapping[key]["seed"], mapping[key]["instruction"]) for mapping in maps}
        if len(identities) != 1:
            raise RuntimeError(f"seed/instruction mismatch for {key}")

    base_curve, parent_curve, pim_curve = map(curve, (base, parent, pim))
    independent_deltas = [
        100 * (pim - base)
        for pim, base in zip(
            pim_curve["independent_success_rates"],
            base_curve["independent_success_rates"],
        )
    ]
    cumulative_delta = pim_curve["cumulative_successes"][3] - base_curve["cumulative_successes"][3]
    payload = {
        "schema": "zeva-pim-user-authorized-formal-three-way-v2",
        "authorization": {
            "user_authorized_after_development_gate_failure": True,
            "development_gate_passed": False,
            "development_pim_minus_parent_attempt4": 2,
            "development_pim_minus_parent_first_attempt": -1,
        },
        "selection_audit": {
            "formal_success_labels_used_for_selection": False,
            "checkpoint_reselected": False,
            "tasks_reselected": False,
            "seeds_reselected": False,
            "gate_changed": False,
        },
        "protocol": {
            "seed_manifest_sha256": manifest_shas[0],
            "tasks": 10,
            "episodes_per_task": 20,
            "episodes": 200,
            "max_attempts": 4,
            "instruction_type": "seen",
            "model_seed_policy": "continuous",
            "action_contract": "eef16_h50_execute_h15",
        },
        "checks": {
            "complete_600_episode_coverage": True,
            "identical_seed_instruction_manifest": True,
            "all_attempt_videos_present": True,
            "reset_scopes_match_protocol": True,
        },
        "conditions": {
            "base": {
                "reset_scope": "episode", "curve": base_curve,
                "per_task": per_task(base), "videos": base_videos,
                "cumulative_successes": base_report["total_successes"],
            },
            "parent": {
                "reset_scope": "episode", "curve": parent_curve,
                "per_task": per_task(parent), "videos": parent_videos,
                "cumulative_successes": parent_report["total_successes"],
            },
            "pim": {
                "reset_scope": "attempt", "curve": pim_curve,
                "per_task": per_task(pim), "videos": pim_videos,
                "cumulative_successes": pim_report["total_successes"],
            },
        },
        "paired": {
            "pim_minus_base": paired_delta(pim, base),
            "parent_minus_base": paired_delta(parent, base),
            "pim_minus_parent": paired_delta(pim, parent),
        },
        "independent_attempt_interpretation": {
            "primary_metric": "successes_at_attempt_divided_by_episodes_reaching_that_attempt",
            "cumulative_success_is_not_an_independent_attempt_rate": True,
            "later_attempt_cohorts_are_survivor_sets": True,
            "pim_minus_base_percentage_points_by_attempt": independent_deltas,
            "meets_plus_4pp_by_attempt": [delta >= 4.0 for delta in independent_deltas],
            "consistent_plus_4pp_across_all_attempts": all(
                delta >= 4.0 for delta in independent_deltas
            ),
            "attempt1_plus_4pp_goal_passed": independent_deltas[0] >= 4.0,
        },
        "cumulative_four_attempt_diagnostic": {
            "comparison": "pim_minus_normally_trained_base_after_up_to_four_attempts",
            "observed_extra_successes": cumulative_delta,
            "observed_percentage_points": 100 * cumulative_delta / 200,
            "not_used_as_independent_attempt_success_rate": True,
        },
        "historical_disclosure": {
            "old_failed_formal": {"base": 111, "old_zeva": 106, "episodes": 200},
            "no_pim_single_attempt_formal": {"base": 111, "parent": 122, "episodes": 200},
            "original_formal_set_had_prior_exposure": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
