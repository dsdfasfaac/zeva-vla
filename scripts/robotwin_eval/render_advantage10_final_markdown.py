#!/usr/bin/env python3
"""Render an audited Anchor/Base/ZeVA RoboTwin result as a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def task_map(report: dict[str, Any], condition: str) -> dict[str, dict[str, Any]]:
    if report.get("condition") != condition:
        raise ValueError(f"expected {condition} report, got {report.get('condition')!r}")
    rows = report.get("tasks")
    if not isinstance(rows, list):
        raise ValueError(f"{condition} report has no task rows")
    result = {str(row["task"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{condition} report contains duplicate tasks")
    return result


def count_cell(row: dict[str, Any]) -> str:
    successes = int(row["successes"])
    episodes = int(row["episodes"])
    rate = float(row["success_rate"])
    if episodes <= 0 or abs(rate - successes / episodes) > 1e-12:
        raise ValueError(f"inconsistent success count: {row}")
    return f"{successes}/{episodes}（{100 * rate:.2f}%）"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="completed paired evaluation root")
    parser.add_argument("--output", type=Path, help="write Markdown here instead of stdout")
    args = parser.parse_args()

    root = args.root.resolve()
    paired = load_json(root / "paired_report.json")
    acceptance = load_json(root / "acceptance.json")
    audit = load_json(root / "completion_audit.json")
    manifest = load_json(root / "manifest.json")
    baseline_report = load_json(root / "baseline" / "report.json")
    zeva_report = load_json(root / "zeva" / "report.json")
    anchor_path = root / "anchor" / "report.json"
    if anchor_path.is_file():
        anchor_report = load_json(anchor_path)
        anchor_is_base = False
    elif manifest.get("baseline_is_untouched_anchor") is True:
        # A precomputed untouched PI Base is itself the anchor.  Keep one
        # authoritative rollout rather than manufacturing a duplicate
        # condition directory merely to satisfy the renderer.
        anchor_report = dict(baseline_report)
        anchor_report["condition"] = "anchor"
        anchor_is_base = True
    else:
        raise FileNotFoundError(f"required result is missing: {anchor_path}")
    condition_reports = {
        "anchor": anchor_report,
        "baseline": baseline_report,
        "zeva": zeva_report,
    }
    conditions = {
        condition: task_map(condition_reports[condition], condition)
        for condition in condition_reports
    }

    if paired.get("schema") != "zeva-robotwin-formal-paired-report-v1":
        raise ValueError("unexpected paired-report schema")
    if acceptance.get("schema") != "zeva-advantage10-acceptance-v1":
        raise ValueError("unexpected acceptance schema")
    if audit.get("schema") != "zeva-advantage10-completion-audit-v1":
        raise ValueError("unexpected completion-audit schema")
    if audit.get("passed") is not True:
        raise ValueError("completion audit did not pass")
    task_names = list(conditions["baseline"])
    if len(task_names) != 10 or any(set(rows) != set(task_names) for rows in conditions.values()):
        raise ValueError("Anchor/Base/ZeVA must contain the same ten tasks")
    totals = {name: int(report["total_episodes"]) for name, report in condition_reports.items()}
    if len(set(totals.values())) != 1 or next(iter(totals.values())) != 200:
        raise ValueError(f"expected 200 episodes per condition, got {totals}")
    if int(paired["total_paired_episodes"]) != 200:
        raise ValueError("paired report does not contain 200 episodes")

    accepted = acceptance.get("accepted") is True
    rates = acceptance["success_rates"]
    lines = [
        "### ZeVA 十任务正式同-seed配对评测",
        "",
        (
            f"协议：`{manifest['camera']}`、`{manifest['instruction_type']}` 指令、"
            f"`{manifest['action_contract']}`、每任务 {manifest['episodes_per_task']} 个 "
            f"expert-valid seeds、模型 RNG `{manifest['model_seed_policy']}`。"
        ),
        "",
        *( ["说明：该次评测的 Base 即未改动 PI Anchor，两列来自同一份经审计的 rollout。", ""]
           if anchor_is_base else [] ),
        "| 条件 | 成功次数 | 成功率 |",
        "|---|---:|---:|",
    ]
    for condition, label in (("anchor", "Untouched PI Anchor"), ("baseline", "Base"), ("zeva", "ZeVA")):
        report = condition_reports[condition]
        lines.append(
            f"| {label} | {int(report['total_successes'])}/200 | "
            f"{100 * float(report['micro_success_rate']):.2f}% |"
        )
    lines += [
        "",
        "| 任务 | Anchor | Base | ZeVA | ZeVA−Base |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in task_names:
        anchor_row = conditions["anchor"][task]
        base_row = conditions["baseline"][task]
        zeva_row = conditions["zeva"][task]
        delta = 100 * (float(zeva_row["success_rate"]) - float(base_row["success_rate"]))
        lines.append(
            f"| `{task}` | {count_cell(anchor_row)} | {count_cell(base_row)} | "
            f"{count_cell(zeva_row)} | {delta:+.2f} pp |"
        )
    ci = paired["paired_bootstrap_95ci"]
    discordant = paired["discordant_pairs"]
    requirements = acceptance["requirements"]
    lines += [
        "",
        f"结论：**{'通过' if accepted else '未通过'}**最终验收。ZeVA−Base 为 "
        f"`{100 * float(paired['absolute_delta']):+.2f} pp`；paired bootstrap 95% CI "
        f"为 `[{100 * float(ci[0]):+.2f}, {100 * float(ci[1]):+.2f}] pp`，exact McNemar "
        f"`p={float(paired['exact_mcnemar_p']):.6g}`。discordant pairs：Base-only "
        f"`{int(discordant['baseline_only'])}`、ZeVA-only `{int(discordant['zeva_only'])}`。",
        "",
        f"硬门槛：Base≥同-seed Anchor `{requirements['baseline_not_below_original_pi_anchor']}`；"
        f"Base≥57% `{requirements['baseline_not_below_historical_normal_pi']}`；"
        f"ZeVA>Base `{requirements['zeva_strictly_above_trained_baseline']}`。"
    ]
    # Cross-check the acceptance summary against the condition reports before emitting.
    for condition in ("anchor", "baseline", "zeva"):
        observed = float(condition_reports[condition]["micro_success_rate"])
        if abs(float(rates[condition]) - observed) > 1e-12:
            raise ValueError(f"acceptance/report rate mismatch for {condition}")

    rendered = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
