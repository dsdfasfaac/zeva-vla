"""Independent structural/provenance audit for cross-attempt PIM pairings."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro

from openpi.zeva.cte_eap_policy import file_sha
from scripts.build_robotwin_pim_artifacts import PIM_ARTIFACT_SCHEMA, record_parts


@dataclasses.dataclass
class Args:
    cte_artifacts: str
    pim_artifacts: str
    output: str


def main(args: Args) -> None:
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite a prior PIM audit.")
    cte = torch.load(args.cte_artifacts, map_location="cpu", weights_only=False)
    pim = torch.load(args.pim_artifacts, map_location="cpu", weights_only=False)
    cte_sha = file_sha(args.cte_artifacts)
    if pim.get("schema") != PIM_ARTIFACT_SCHEMA or pim.get("cte_artifacts_sha256") != cte_sha:
        raise ValueError("PIM/CTE artifact lineage mismatch.")
    train = cte["splits"]["train"]
    validation = cte["splits"]["validation"]
    if set(train) & set(validation):
        raise ValueError("CTE train/validation overlap.")
    mode_counts = {name: 0 for name in ("matched", "same_condition_far", "cross_condition")}
    for split, targets in (("train", train), ("validation", validation)):
        rows = pim["pairings"].get(split, {})
        if set(rows) != set(targets):
            raise ValueError(f"PIM {split} target coverage is incomplete.")
        for target_id, row in rows.items():
            target_condition, target_task, _ = record_parts(target_id)
            query = F.normalize(targets[target_id]["phase"][0].float(), dim=-1)
            for mode in mode_counts:
                source_id = row[mode]
                if source_id not in train or source_id == target_id:
                    raise ValueError("PIM source is not a distinct train-only episode.")
                condition, task, _ = record_parts(source_id)
                if task != target_task:
                    raise ValueError("PIM source crosses tasks.")
                if mode == "cross_condition" and condition == target_condition:
                    raise ValueError("Cross-condition PIM source stayed in the target condition.")
                if mode != "cross_condition" and condition != target_condition:
                    raise ValueError("Same-condition PIM source crossed conditions.")
                score = float(query @ F.normalize(train[source_id]["phase"][0].float(), dim=-1))
                if abs(score - float(row["cosine"][mode])) > 1e-6:
                    raise ValueError("Stored PIM cosine does not match frozen CTE features.")
                mode_counts[mode] += 1
    report = {
        "schema": "zeva-pim-pairing-audit-v1", "status": "PASS",
        "cte_artifacts_sha256": cte_sha,
        "pim_artifacts_sha256": file_sha(args.pim_artifacts),
        "source_subset": "train-only", "success_labels_used": False,
        "train_targets": len(train), "validation_targets": len(validation),
        "mode_counts": mode_counts, "source_sha256": file_sha(__file__),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
