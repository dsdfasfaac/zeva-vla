#!/usr/bin/env python3
"""Audit a completed 10-task Base/ZeVA or Base/Anchor/ZeVA evaluation.

The launcher performs checks at the end of each condition.  This independent
audit deliberately re-derives the final numbers from per-episode progress and
video files, and verifies that Base, the untouched PI0.5 anchor, and ZeVA used
the exact same sparse expert-valid seeds and language instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


VIDEO_PATTERN = re.compile(
    r"^episode(?P<index>\d+)_randomized-true_success-(?P<success>true|false)\.mp4$"
)
EXPECTED_MANIFEST = {
    "instruction_type": "seen",
    "camera": "Large_D435_640x480",
    "action_contract": "chunk-start-relative-eef16-predict-h50-execute-h15",
    "model_seed_policy": "continuous",
    "model_rng_seed": 20260907,
    "model_runtime": "native_handoff_transformers_5.5.4",
}
EXPECTED_IDENTITY = {
    "task_config": "zeva_randomized",
    "instruction_type": "seen",
    "execute_horizon": 15,
    "model_seed_policy": "continuous",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Audit JSON destination (default: ROOT/completion_audit.json)",
    )
    parser.add_argument(
        "--require-accepted",
        action="store_true",
        help="Also require Base >= Anchor and ZeVA > Base.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "completion_audit.json").resolve()
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    tasks_path = root / "tasks.txt"
    manifest_path = root / "manifest.json"
    seed_manifest_path = root / "seed_manifest.json"
    check(tasks_path.is_file(), "missing tasks.txt")
    check(manifest_path.is_file(), "missing manifest.json")
    check(seed_manifest_path.is_file(), "missing seed_manifest.json")
    if errors:
        payload = {"schema": "zeva-advantage10-completion-audit-v1", "passed": False,
                   "errors": errors, "root": str(root)}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 1

    tasks = [line.strip() for line in tasks_path.read_text().splitlines() if line.strip()]
    manifest = load_json(manifest_path)
    seed_manifest = load_json(seed_manifest_path)
    expected_task_count = int(manifest.get("task_count", -1))
    expected_episodes = int(manifest.get("episodes_per_task", -1))
    expected_total = expected_task_count * expected_episodes
    expected_start_seed = int(manifest.get("absolute_start_seed", -1))
    expected_model_rng_seed = int(manifest.get("model_rng_seed", -1))
    baseline_is_untouched_anchor = bool(manifest.get("baseline_is_untouched_anchor", False))
    conditions = (("baseline", "zeva") if baseline_is_untouched_anchor
                  else ("baseline", "anchor", "zeva"))
    check(expected_task_count == 10, f"formal task_count must be 10, got {expected_task_count}")
    check(expected_episodes > 0, f"episodes_per_task must be positive, got {expected_episodes}")
    check(expected_start_seed >= 0, f"absolute_start_seed must be non-negative, got {expected_start_seed}")
    check(expected_model_rng_seed == 20260907,
          f"model_rng_seed must be 20260907, got {expected_model_rng_seed}")
    check(len(tasks) == expected_task_count,
          f"expected {expected_task_count} tasks, got {len(tasks)}")
    check(len(tasks) == len(set(tasks)), "tasks.txt contains duplicates")
    for key, expected in EXPECTED_MANIFEST.items():
        check(manifest.get(key) == expected,
              f"manifest.{key}: expected {expected!r}, got {manifest.get(key)!r}")
    check(seed_manifest.get("schema") == "robotwin-expert-valid-seeds-v1",
          "invalid seed manifest schema")
    check(seed_manifest.get("start_seed") == expected_start_seed,
          f"seed manifest must start at {expected_start_seed}")
    check(seed_manifest.get("episodes_per_task") == expected_episodes,
          f"seed manifest must contain {expected_episodes} episodes per task")
    seed_tasks = seed_manifest.get("tasks", {})
    check(set(seed_tasks) == set(tasks), "seed manifest task set differs from tasks.txt")

    condition_summaries: dict[str, dict[str, Any]] = {}
    episode_tables: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for condition in conditions:
        condition_root = root / condition
        state_path = condition_root / "state.json"
        report_path = condition_root / "report.json"
        check(state_path.is_file(), f"{condition}: missing state.json")
        check(report_path.is_file(), f"{condition}: missing report.json")
        if state_path.is_file():
            check(load_json(state_path).get("state") == "complete",
                  f"{condition}: condition state is not complete")
        report = load_json(report_path) if report_path.is_file() else {}
        check(report.get("condition") == condition,
              f"{condition}: report condition mismatch")
        check(report.get("task_count") == expected_task_count,
              f"{condition}: report task_count != {expected_task_count}")
        check(report.get("episodes_per_task") == expected_episodes,
              f"{condition}: report episodes_per_task != {expected_episodes}")
        check(report.get("total_episodes") == expected_total,
              f"{condition}: report total_episodes != {expected_total}")
        check(report.get("profile") == "randomized_seen_large_d435",
              f"{condition}: report profile mismatch")
        check(report.get("action_contract") == "eef16_h50_execute_h15",
              f"{condition}: report action contract mismatch")
        check(report.get("model_seed_policy") == "continuous",
              f"{condition}: report model seed policy mismatch")

        total_successes = 0
        total_videos = 0
        task_rows: list[dict[str, Any]] = []
        episode_tables[condition] = {}
        report_rows = {row["task"]: row for row in report.get("tasks", [])}
        check(set(report_rows) == set(tasks), f"{condition}: report task set mismatch")
        for task in tasks:
            progress_path = condition_root / "progress" / f"{task}.json"
            status_path = condition_root / "status" / f"{task}.json"
            check(progress_path.is_file(), f"{condition}/{task}: missing progress")
            check(status_path.is_file(), f"{condition}/{task}: missing status")
            if not progress_path.is_file():
                continue
            progress = load_json(progress_path)
            identity = progress.get("identity", {})
            for key, expected in EXPECTED_IDENTITY.items():
                check(identity.get(key) == expected,
                      f"{condition}/{task}: identity.{key} mismatch")
            check(identity.get("absolute_start_seed") == expected_start_seed,
                  f"{condition}/{task}: identity.absolute_start_seed mismatch")
            check(identity.get("target_episodes") == expected_episodes,
                  f"{condition}/{task}: identity.target_episodes mismatch")
            check(identity.get("task_name") == task,
                  f"{condition}/{task}: identity task mismatch")
            replayed_frozen_manifest = manifest.get("seed_selection") == (
                "all-conditions-replay-existing-frozen-expert-valid-manifest"
            )
            expected_fixed = replayed_frozen_manifest or condition != "baseline"
            check(identity.get("fixed_seed_sequence") is expected_fixed,
                  f"{condition}/{task}: fixed_seed_sequence should be {expected_fixed}")
            episodes = progress.get("episode_results", [])
            check(progress.get("complete") is True, f"{condition}/{task}: incomplete")
            check(progress.get("completed_episodes") == expected_episodes,
                  f"{condition}/{task}: completed_episodes != {expected_episodes}")
            check(len(episodes) == expected_episodes,
                  f"{condition}/{task}: episode list length != {expected_episodes}")
            indices = [item.get("episode_index") for item in episodes]
            check(indices == list(range(expected_episodes)),
                  f"{condition}/{task}: invalid episode indices")
            seeds = [int(item["seed"]) for item in episodes]
            instructions = [str(item["instruction"]) for item in episodes]
            check(all(seed >= expected_start_seed for seed in seeds),
                  f"{condition}/{task}: seed below {expected_start_seed}")
            check(len(seeds) == len(set(seeds)), f"{condition}/{task}: duplicate seeds")
            expected_pairs = [
                (int(item["seed"]), str(item["instruction"]))
                for item in seed_tasks.get(task, [])
            ]
            check(list(zip(seeds, instructions, strict=True)) == expected_pairs,
                  f"{condition}/{task}: seed/instruction differs from frozen manifest")
            if condition == "baseline":
                check(all(right > left for left, right in zip(seeds, seeds[1:])),
                      f"{condition}/{task}: expert-valid seeds are not strictly increasing")

            successes = sum(bool(item["success"]) for item in episodes)
            check(progress.get("successes") == successes,
                  f"{condition}/{task}: progress success total mismatch")
            result_root = condition_root / "results" / task
            videos: dict[int, bool] = {}
            unexpected_videos: list[str] = []
            for video in result_root.glob("*.mp4"):
                match = VIDEO_PATTERN.match(video.name)
                if match is None:
                    unexpected_videos.append(video.name)
                    continue
                check(video.stat().st_size > 0, f"{condition}/{task}: empty video {video.name}")
                index = int(match.group("index"))
                check(index not in videos, f"{condition}/{task}: duplicate video index {index}")
                videos[index] = match.group("success") == "true"
            check(not unexpected_videos,
                  f"{condition}/{task}: unfinished/unexpected videos {unexpected_videos}")
            check(set(videos) == set(range(expected_episodes)),
                  f"{condition}/{task}: final video indices are incomplete")
            for item in episodes:
                index = int(item["episode_index"])
                check(videos.get(index) == bool(item["success"]),
                      f"{condition}/{task}: video/progress mismatch at episode {index}")
            if status_path.is_file():
                check(load_json(status_path).get("return_code") == 0,
                      f"{condition}/{task}: worker return_code != 0")
            report_row = report_rows.get(task, {})
            check(report_row.get("successes") == successes,
                  f"{condition}/{task}: report success total mismatch")
            check(report_row.get("seeds") == seeds,
                  f"{condition}/{task}: report seed list mismatch")

            total_successes += successes
            total_videos += len(videos)
            episode_tables[condition][task] = episodes
            task_rows.append({"task": task, "successes": successes,
                              "success_rate": successes / expected_episodes,
                              "videos": len(videos)})

        check(total_videos == expected_total,
              f"{condition}: expected {expected_total} final videos, got {total_videos}")
        check(report.get("total_successes") == total_successes,
              f"{condition}: report total_successes mismatch")
        expected_rate = total_successes / expected_total
        check(report.get("micro_success_rate") == expected_rate,
              f"{condition}: report micro rate mismatch")
        condition_summaries[condition] = {
            "episodes": expected_total,
            "successes": total_successes,
            "success_rate": expected_rate,
            "videos": total_videos,
            "tasks": task_rows,
        }

    # Cross-condition pairing must include both the environment seed and the
    # sampled seen-language instruction, not just aggregate task counts.
    if all(condition in episode_tables for condition in conditions):
        for task in tasks:
            if not all(task in episode_tables[c] for c in conditions):
                continue
            pairs = []
            for condition in conditions:
                pairs.append([
                    (int(item["seed"]), str(item["instruction"]))
                    for item in episode_tables[condition][task]
                ])
            check(all(pair == pairs[0] for pair in pairs[1:]),
                  f"{task}: evaluation conditions are not exactly paired")

    paired_path = root / "paired_report.json"
    acceptance_path = root / "acceptance.json"
    check(paired_path.is_file(), "missing paired_report.json")
    check(acceptance_path.is_file(), "missing acceptance.json")
    paired = load_json(paired_path) if paired_path.is_file() else {}
    acceptance = load_json(acceptance_path) if acceptance_path.is_file() else {}
    if all(name in condition_summaries for name in conditions):
        baseline_rate = condition_summaries["baseline"]["success_rate"]
        anchor_rate = (baseline_rate if baseline_is_untouched_anchor else
                       condition_summaries["anchor"]["success_rate"])
        zeva_rate = condition_summaries["zeva"]["success_rate"]
        check(paired.get("total_paired_episodes") == expected_total,
              f"paired report does not contain {expected_total} episodes")
        expected_paired_anchor = None if baseline_is_untouched_anchor else anchor_rate
        check(paired.get("anchor_success_rate") == expected_paired_anchor,
              "paired report anchor rate mismatch")
        check(paired.get("baseline_success_rate") == baseline_rate,
              "paired report baseline rate mismatch")
        check(paired.get("zeva_success_rate") == zeva_rate,
              "paired report ZeVA rate mismatch")
        check(paired.get("absolute_delta") == zeva_rate - baseline_rate,
              "paired report Base/ZeVA delta mismatch")
        minimum_baseline_rate = float(manifest.get("min_baseline_success_rate", 0.0))
        baseline_floor = max(anchor_rate, minimum_baseline_rate)
        accepted = baseline_rate >= baseline_floor and zeva_rate > baseline_rate
        check(acceptance.get("accepted") is accepted,
              "acceptance.json disagrees with recomputed criteria")
        check(
            acceptance.get("minimum_baseline_success_rate") == minimum_baseline_rate,
            "acceptance.json historical normal-PI floor mismatch",
        )
        check(
            acceptance.get("effective_baseline_floor") == baseline_floor,
            "acceptance.json effective baseline floor mismatch",
        )
        if args.require_accepted:
            check(
                accepted,
                "acceptance failed: require Base >= max(Anchor, historical normal PI) "
                "and ZeVA > Base",
            )

    config_evidence = {}
    config_keys = (["baseline_config", "zeva_config"] if baseline_is_untouched_anchor
                   else ["baseline_config", "anchor_config", "zeva_config"])
    for key in config_keys:
        value = manifest.get(key)
        path = Path(value) if value else None
        check(path is not None and path.is_file(), f"manifest {key} is missing or unreadable")
        if path is not None and path.is_file():
            config_evidence[key] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

    if manifest.get("seed_selection") == (
        "all-conditions-replay-existing-frozen-expert-valid-manifest"
    ):
        source_value = manifest.get("seed_manifest_source")
        source = Path(source_value) if source_value else None
        check(source is not None and source.is_file(),
              "frozen seed manifest source is missing or unreadable")
        if source is not None and source.is_file():
            source_digest = sha256(source)
            check(manifest.get("seed_manifest_source_sha256") == source_digest,
                  "frozen seed manifest source hash mismatch")
            check(sha256(seed_manifest_path) == source_digest,
                  "copied seed manifest differs from frozen source")

    if baseline_is_untouched_anchor and manifest.get("precomputed_baseline_root"):
        source_value = manifest.get("precomputed_baseline_root")
        source = Path(source_value) if source_value else None
        source_report = source / "report.json" if source is not None else None
        check(source_report is not None and source_report.is_file(),
              "precomputed untouched PI baseline source report is missing")
        if source_report is not None and source_report.is_file():
            check(manifest.get("precomputed_baseline_report_sha256") == sha256(source_report),
                  "precomputed untouched PI baseline report hash mismatch")
        baseline_summary = condition_summaries.get("baseline", {})
        expected_successes = int(manifest.get("precomputed_baseline_expected_successes", -1))
        check(baseline_summary.get("successes") == expected_successes,
              "imported untouched PI baseline differs from its preregistered count")

    payload = {
        "schema": "zeva-advantage10-completion-audit-v1",
        "passed": not errors,
        "require_accepted": args.require_accepted,
        "root": str(root),
        "task_count": len(tasks),
        "conditions": condition_summaries,
        "config_evidence": config_evidence,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"passed": not errors, "errors": errors,
                      "output": str(output)}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
