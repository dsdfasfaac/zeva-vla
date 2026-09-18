"""Independent raw-record audit before the fixed full-PI Stage2 run.

The exported recurrent cache and memory are checked against the raw adapter,
not against the exporter index alone. A PASS report is tied to file hashes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from openpi.zeva.behavior_effect import SCHEMA
from openpi.zeva.behavior_effect_policy import file_sha
from openpi.zeva.robotwin_contract import RobotWinHandoff
from scripts.behavior_effect_validation_contract import check_cache_coverage, enumerate_decisions
from scripts.train_robotwin_zte import TorchCodecRoboTwinDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "artifacts", "retrieval", "dataset-root", "foundation", "handoff", "tasks", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--exploratory-epoch40", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tasks = json.loads(args.tasks.read_text())["task_names"]
    if len(tasks) != 10 or len(set(tasks)) != 10:
        raise ValueError("Audit requires the frozen ten unique tasks.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected_epoch = 40 if args.exploratory_epoch40 else 80
    if (checkpoint.get("schema") != SCHEMA or checkpoint.get("epoch") != expected_epoch
            or not checkpoint["manifest"]["promotable"]
            or not (checkpoint.get("validation") or {}).get("stage1_gate")):
        raise ValueError(f"Epoch{expected_epoch} Stage1 validation gate did not pass.")
    manifest = checkpoint["manifest"]
    if (manifest["tasks"] != tasks or manifest["train_episodes"] != 5230
            or manifest["validation_episodes"] != 270
            or checkpoint["step"] != expected_epoch * 654
            or manifest["normalization"] != "baseline-mean-std"
            or manifest["decoder"] != "torchcodec"
            or manifest["source_sha256"]["trainer"] != file_sha(Path(__file__).with_name(
                "train_robotwin_behavior_effect_cte.py"))
            or manifest["source_sha256"]["model"] != file_sha(Path(__file__).parents[1] /
                "src/openpi/zeva/behavior_effect.py")):
        raise ValueError("Stage1 task/count/source/normalization contract differs.")
    adapter = args.dataset_root / "adapter.json"
    stats = RobotWinHandoff.from_root(args.handoff).statistics
    if manifest["adapter_sha256"] != file_sha(adapter) or manifest["statistics_sha256"] != file_sha(stats):
        raise ValueError("Stage1 adapter/statistics differ from current sources.")
    identity_evidence = manifest.get("dataset_identity") or {}
    for label, path_key, hash_key in (("replica", "replica", "replica_sha256"),
                                      ("canonical", "reference", "reference_sha256")):
        path = identity_evidence.get(path_key)
        if not path or file_sha(path) != identity_evidence[hash_key]:
            raise ValueError(f"Stage1 {label} source evidence changed or is absent.")
    if (identity_evidence["reference_sha256"] !=
            "3ba76d89295564cabb735aecc47ddb55cb039fe21f4072138a14d1f02fb20666"):
        raise ValueError("Canonical source report is no longer the frozen report.")
    if not identity_evidence.get("task_source_sha256"):
        raise ValueError("Independent selected-task canonical source proof absent.")
    if file_sha(manifest["args"]["source_task_identity_report"]) != identity_evidence["task_source_sha256"]:
        raise ValueError("Independent task source proof changed.")
    artifact = torch.load(args.artifacts, map_location="cpu", weights_only=False)
    if (artifact.get("schema") != SCHEMA + "-artifacts" or artifact["cte_sha256"] != file_sha(args.checkpoint)
            or artifact["adapter_sha256"] != file_sha(adapter) or artifact["tasks"] != tasks
            or artifact["bank_subset"] != "train" or artifact["execution_horizon"] != 15
            or artifact["policy_horizon"] != 50
            or bool(artifact.get("exploratory_epoch40", False)) != args.exploratory_epoch40):
        raise ValueError("Exported artifact lineage/schema/protocol differs.")
    retrieval = torch.load(args.retrieval, map_location="cpu", weights_only=False)
    if retrieval.get("source_feature") != "task_language" or not set(tasks).issubset(retrieval["task_names"]):
        raise ValueError("Frozen task-language retrieval is incompatible with this memory.")
    rows = {}
    train_episode_tasks = None
    for split, expected_episodes in (("train", 5230), ("validation", 270)):
        source = TorchCodecRoboTwinDataset(adapter, split)
        decisions = enumerate_decisions(source.dataset._records, tasks)
        records = {row["record_id"] for row in decisions}
        raw_episode_keys = sorted((record["key"][0], record["key"][1], int(record["episode_index"]))
                                  for record in source.dataset._records
                                  if record["key"][1] in tasks and int(record["length"]) > 15)
        if len(records) != expected_episodes or len(records) != len(artifact["splits"][split]):
            raise ValueError(f"Raw {split} episode count/coverage differs.")
        if raw_episode_keys != manifest[f"{split}_keys"]:
            raise ValueError(f"Raw {split} trajectory IDs differ from fixed Stage1 manifest.")
        if split == "train":
            train_episode_tasks = {f"{record['key'][0]}:{record['key'][1]}:{record['episode_index']}":record["key"][1]
                                   for record in source.dataset._records
                                   if record["key"][1] in tasks and int(record["length"]) > 15}
        check_cache_coverage(decisions, artifact["splits"][split])
        for value in artifact["splits"][split].values():
            if not torch.isfinite(value["phase"]).all() or not torch.isfinite(value["effect"]).all():
                raise ValueError(f"Non-finite {split} cache feature.")
        counts = Counter(row["task"] for row in decisions)
        rows[split] = {"episodes":len(records), "decisions":len(decisions),
                       "per_task_decisions":dict(sorted(counts.items()))}
    train_ids = set(artifact["splits"]["train"])
    validation_ids = set(artifact["splits"]["validation"])
    bank_ids = artifact["bank_record_ids"]
    if (train_ids & validation_ids or len(bank_ids) != len(set(bank_ids))
            or set(bank_ids) != train_ids or artifact["keys"].shape != (5230, 128)
            or artifact["values"].shape != (5230, 256)
            or artifact["task_ids"].shape != (5230,)
            or not torch.isfinite(artifact["keys"]).all()
            or not torch.isfinite(artifact["values"]).all()):
        raise ValueError("Bank/cache train-only, shape or finite checks failed.")
    for i, record_id in enumerate(bank_ids):
        if not 0 <= int(artifact["task_ids"][i]) < len(tasks) or tasks[int(artifact["task_ids"][i])] != train_episode_tasks[record_id]:
            raise ValueError(f"Bank task id misassigned: {record_id}")
    foundation_model = args.foundation / "model.safetensors"
    if not foundation_model.is_file():
        raise ValueError("Foundation best-v1 weights absent.")
    if file_sha(foundation_model) != "7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe":
        raise ValueError("Foundation best-v1 SHA differs from frozen selected handoff.")
    report = {"schema":"zeva-behavior-effect-prestage2-audit-v1", "status":"PASS",
              "stage1_epoch":expected_epoch, "exploratory_epoch40":args.exploratory_epoch40,
              "formal_promotion_eligible":not args.exploratory_epoch40,
              "stage1_step":checkpoint["step"],
              "validation":checkpoint["validation"], "split":rows,
              "train_bank_entries":len(bank_ids), "validation_in_bank":0,
              "sha256":{"checkpoint":file_sha(args.checkpoint), "artifact":file_sha(args.artifacts),
                        "retrieval":file_sha(args.retrieval), "adapter":file_sha(adapter),
                        "statistics":file_sha(stats), "foundation":file_sha(foundation_model),
                        "audit_script":file_sha(__file__)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status":"PASS", "split":rows, "output":str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
