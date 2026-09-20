#!/usr/bin/env python3
"""Independently audit the frozen single-attempt Episode-PIM development run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = (
    "3160b039b7c571850ef8187f61818c0aaeaa0a8ebc88c8c189b1cd8fcdc86103"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_condition(
    root: Path, condition_dir: str
) -> tuple[dict[tuple[str, int, str], dict], dict, int]:
    condition_root = root / condition_dir
    report = json.loads((condition_root / "report.json").read_text())
    if (
        report.get("total_episodes") != 80
        or report.get("task_count") != 10
        or report.get("episodes_per_task") != 8
        or report.get("total_attempts_executed") != 80
    ):
        raise RuntimeError(f"{root}: expected complete single-attempt 10x8 report")

    rows: dict[tuple[str, int, str], dict] = {}
    for progress in sorted((condition_root / "progress").glob("*.json")):
        payload = json.loads(progress.read_text())
        episodes = payload.get("episode_results", [])
        identity = payload.get("identity", {})
        if not payload.get("complete") or len(episodes) != 8:
            raise RuntimeError(f"{progress}: incomplete")
        if identity.get("max_policy_attempts") != 1:
            raise RuntimeError(f"{progress}: not a single-attempt evaluation")
        if identity.get("attempt_reset_scope") != "episode":
            raise RuntimeError(f"{progress}: wrong reset scope")
        task = progress.stem
        for episode in episodes:
            attempts = episode.get("attempts", [])
            if len(attempts) != 1 or int(attempts[0].get("attempt", 1)) != 1:
                raise RuntimeError(f"{progress}: invalid attempt trace")
            if bool(episode["success"]) != bool(attempts[0]["success"]):
                raise RuntimeError(f"{progress}: episode/attempt outcome mismatch")
            key = (task, int(episode["seed"]), str(episode["instruction"]))
            if key in rows:
                raise RuntimeError(f"{progress}: duplicate episode identity {key}")
            rows[key] = episode
    if len(rows) != 80:
        raise RuntimeError(f"{root}: only {len(rows)} episode traces")

    videos = list((condition_root / "results").glob("**/*success-*.mp4"))
    if len(videos) != 80:
        raise RuntimeError(f"{root}: expected 80 attempt videos, got {len(videos)}")
    successes = sum(bool(row["success"]) for row in rows.values())
    if report.get("total_successes") != successes:
        raise RuntimeError(f"{root}: report success count mismatch")
    return rows, report, len(videos)


def paired_delta(left: dict, right: dict) -> dict:
    left_only = right_only = both_success = both_failure = 0
    for key in sorted(left):
        left_success = bool(left[key]["success"])
        right_success = bool(right[key]["success"])
        if left_success and not right_success:
            left_only += 1
        elif right_success and not left_success:
            right_only += 1
        elif left_success:
            both_success += 1
        else:
            both_failure += 1
    return {
        "left_only_successes": left_only,
        "right_only_successes": right_only,
        "net_successes": left_only - right_only,
        "percentage_points": 100 * (left_only - right_only) / len(left),
        "both_success": both_success,
        "both_failure": both_failure,
    }


def per_task(rows: dict) -> dict[str, dict]:
    tasks = sorted({key[0] for key in rows})
    output = {}
    for task in tasks:
        task_rows = [value for key, value in rows.items() if key[0] == task]
        successes = sum(bool(row["success"]) for row in task_rows)
        output[task] = {
            "episodes": len(task_rows),
            "successes": successes,
            "success_rate": successes / len(task_rows),
        }
    return output


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

    audit = json.loads((args.base_root / "development_seed_audit.json").read_text())
    if audit.get("status") != "PASS" or any(
        audit.get(key) != 0
        for key in (
            "overlap_with_formal",
            "overlap_with_prior_development",
            "overlap_with_prior_pim_development",
        )
    ):
        raise RuntimeError("development seed audit is not PASS")

    base, base_report, base_videos = load_condition(args.base_root, "baseline")
    parent, parent_report, parent_videos = load_condition(args.parent_root, "zeva")
    pim, pim_report, pim_videos = load_condition(args.pim_root, "zeva")
    if not (base.keys() == parent.keys() == pim.keys()):
        raise RuntimeError("condition seed/instruction identities differ")

    base_successes = int(base_report["total_successes"])
    parent_successes = int(parent_report["total_successes"])
    pim_successes = int(pim_report["total_successes"])
    payload = {
        "schema": "zeva-episode-pim-singleattempt-development-three-way-v1",
        "candidate_status": "post-formal-development-only",
        "protocol": {
            "seed_manifest_sha256": manifest_shas[0],
            "selection_rule": "first-eight-expert-valid-per-task-from-3000000",
            "tasks": 10,
            "episodes_per_task": 8,
            "episodes": 80,
            "max_attempts": 1,
            "instruction_type": "seen",
            "model_seed_policy": "continuous",
            "action_contract": "eef16_h50_execute_h15",
        },
        "checks": {
            "complete_240_episode_coverage": True,
            "identical_seed_instruction_manifest": True,
            "all_240_attempt_videos_present": True,
            "single_attempt_episode_reset_protocol": True,
            "zero_overlap_with_three_prior_sets": True,
        },
        "conditions": {
            "base": {
                "successes": base_successes,
                "success_rate": base_successes / 80,
                "videos": base_videos,
                "per_task": per_task(base),
            },
            "parent": {
                "successes": parent_successes,
                "success_rate": parent_successes / 80,
                "videos": parent_videos,
                "per_task": per_task(parent),
            },
            "episode_pim": {
                "successes": pim_successes,
                "success_rate": pim_successes / 80,
                "videos": pim_videos,
                "per_task": per_task(pim),
            },
        },
        "paired": {
            "episode_pim_minus_base": paired_delta(pim, base),
            "parent_minus_base": paired_delta(parent, base),
            "episode_pim_minus_parent": paired_delta(pim, parent),
        },
        "fixed_gate": {
            "criterion": "episode_pim_successes_strictly_exceed_base_and_parent",
            "episode_pim_gt_base": pim_successes > base_successes,
            "episode_pim_gt_parent": pim_successes > parent_successes,
            "passed": pim_successes > max(base_successes, parent_successes),
        },
        "historical_boundary": {
            "formal_labels_used_for_selection": False,
            "can_revise_prior_formal_conclusions": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
