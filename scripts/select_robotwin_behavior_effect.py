"""Fail-closed validation5 gate for the fixed step5000 CTE+effect candidate.

Consumes per-decision matched-noise rollout-action errors, not training loss,
formal success labels or a partial-validation summary. Expected decision ids
must be exported independently from the validation dataset/cache enumeration.
"""
from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path


CONDITIONS = ("base", "aligned", "within_task_shuffled", "effect_off")


def select(report, expected, plan, tasks):
    if report.get("schema") != "zeva-behavior-effect-validation5-v1":
        raise ValueError("Require a CTE+effect validation5 per-decision report.")
    if report.get("split") != "validation" or report.get("formal_labels_used") is not False:
        raise ValueError("Only untouched-by-formal-labels validation data is eligible.")
    if report.get("checkpoint_step") != 5000 or plan["stage2"]["selection"] != "step5000":
        raise ValueError("Only the preregistered fixed endpoint is eligible.")
    if report.get("output_horizon") != 50 or report.get("execution_horizon") != 15:
        raise ValueError("H50-output/H15-action-MSE contract changed.")
    if report.get("metric") != "sample_mean_normalized_executed_h15_action_mse":
        raise ValueError("Flow error/NLL cannot substitute for sampled H15 action error.")
    if report.get("matched_noise") is not True:
        raise ValueError("All conditions must use identical diffusion noise per decision.")
    if report.get("baseline_model_sha256") != plan["comparison_base_model_sha256"]:
        raise ValueError("Comparison must use the frozen normally trained Base1000 weights.")
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Expected validation decision identifiers must be complete and unique.")
    expected_set = set(expected)
    rows = report["rows"]
    observed = [row["sample_id"] for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected_set:
        raise ValueError("Validation coverage mismatch or duplicate decision identifiers.")
    shuffled = [row["shuffled_sample_id"] for row in rows]
    if len(shuffled) != len(set(shuffled)) or set(shuffled) != expected_set:
        raise ValueError("The alignment control must be a complete permutation, not a repeated exemplar.")
    grouped = defaultdict(list)
    for row in rows:
        if row["task"] not in tasks or not 1 <= row["valid_action_steps"] <= 15:
            raise ValueError("Invalid task or executed-prefix validity mask.")
        if row["shuffled_sample_id"] == row["sample_id"] or row["shuffled_sample_id"] not in expected_set:
            raise ValueError("Shuffling must use a different validation decision.")
        for condition in CONDITIONS:
            value = row[condition]
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {condition} action MSE.")
        grouped[row["task"]].append(row)
    if set(grouped) != set(tasks) or len(tasks) != 10:
        raise ValueError("All ten frozen tasks must be covered.")
    row_by_id = {row["sample_id"]:row for row in rows}
    for row in rows:
        if row_by_id[row["shuffled_sample_id"]]["task"] != row["task"]:
            raise ValueError("A between-task shuffle is not the declared alignment ablation.")
    per_task = {task:{"count":len(entries), **{condition:sum(x[condition] for x in entries)/len(entries)
                                               for condition in CONDITIONS}}
                for task,entries in sorted(grouped.items())}
    average = {condition:sum(row[condition] for row in rows)/len(rows) for condition in CONDITIONS}
    gain = 1-average["aligned"]/average["base"] if average["base"] > 0 else None
    nonworse = sum(row["aligned"] <= row["base"] for row in per_task.values())
    gates = {
        "base_relative_gain": gain is not None and gain >= plan["gates"]["validation_h15_mse_relative_gain"],
        "nonworse_tasks": nonworse >= plan["gates"]["validation_nonworse_tasks"],
        "aligned_beats_within_task_shuffle": average["aligned"] < average["within_task_shuffled"],
        "effect_not_worse_than_removal": average["aligned"] <= average["effect_off"],
    }
    passed = all(gates.values())
    return {"schema":"zeva-behavior-effect-validation5-gate-v1", "passed":passed,
            "selected_checkpoint":report["checkpoint"] if passed else None,
            "selected_step":5000 if passed else None, "formal_labels_used":False,
            "validation_decisions":len(rows), "average":average, "relative_gain":gain,
            "nonworse_tasks":nonworse, "checks":gates, "per_task":per_task,
            "next": "disjoint_development_pair" if passed else "no_closed_loop_promotion"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "expected-decisions", "plan", "tasks", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    checkpoint = Path(report["checkpoint"])
    for filename in ("model.safetensors", "zeva_adapter.pth", "training_state.pth", "COMPLETE"):
        if not (checkpoint/filename).is_file():
            raise ValueError(f"Missing checkpoint component: {filename}")
    if sha(checkpoint/"model.safetensors") != report["model_sha256"]:
        raise ValueError("Validation report belongs to different PI weights.")
    if sha(checkpoint/"zeva_adapter.pth") != report["adapter_sha256"]:
        raise ValueError("Validation report belongs to a different PBD adapter.")
    import torch
    state = torch.load(checkpoint/"training_state.pth", map_location="cpu", weights_only=False)
    if state["step"] != 5000 or state["manifest"]["global_batch"] != 256:
        raise ValueError("Checkpoint step/batch differs from the predeclared run.")
    train_args = state["manifest"]["args"]
    denominator = train_args["batch_size"] * train_args["accumulation"]
    if denominator <= 0 or 256 % denominator:
        raise ValueError("Checkpoint per-rank batch/accumulation is incompatible with global256.")
    for rank in range(256 // denominator):
        if not (checkpoint/f"rng_rank{rank}.pth").is_file():
            raise ValueError(f"Incomplete checkpoint: rank{rank} RNG state absent.")
    result = select(report, json.loads(args.expected_decisions.read_text()), json.loads(args.plan.read_text()),
                    json.loads(args.tasks.read_text())["task_names"])
    result["report_sha256"] = sha(args.report)
    result["expected_decisions_sha256"] = sha(args.expected_decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
