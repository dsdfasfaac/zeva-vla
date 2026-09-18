"""Freeze validation5 decision IDs directly from raw adapter episode metadata.

This runs before Stage2 and does not import CTE, its cache, or model outputs.
It deliberately implements the H15 boundary enumeration separately from the
validation report generator, so complete coverage has a second computation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "tasks", "stage1-manifest", "output", "audit-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit_output.exists():
        raise FileExistsError("Frozen expected IDs/audit already exist.")
    # Import only the metadata adapter. The Stage1 training wrapper also
    # imports Mamba/Triton and requires a visible GPU even for read-only IDs.
    from egoscale.data.robotwin.lerobot import RoboTwinLeRobotEEF16Dataset
    tasks = json.loads(args.tasks.read_text())["task_names"]
    if len(tasks) != 10 or len(set(tasks)) != 10:
        raise ValueError("Only the frozen ten tasks are eligible.")
    manifest = json.loads(args.stage1_manifest.read_text())
    adapter = args.dataset_root / "adapter.json"
    if (manifest["tasks"] != tasks or manifest["validation_episodes"] != 270
            or manifest["adapter_sha256"] != digest(adapter)
            or manifest["selection"] != "fixed-epoch80; validation5-only gate; no formal labels"):
        raise ValueError("Raw adapter and frozen Stage1 validation split differ.")
    source = RoboTwinLeRobotEEF16Dataset(adapter, subset="validation")
    ids, episode_keys, per_task = [], [], Counter()
    for record in source._records:
        subset, task = record["key"]
        length = int(record["length"])
        if task not in tasks or length <= 15:
            continue
        episode = int(record["episode_index"])
        episode_keys.append([subset, task, episode])
        prefix = f"{subset}:{task}:{episode}@"
        # Independent copy of the baseline controller's 0,15,... boundaries.
        starts = [frame for frame in range(0, length, 15) if frame + 15 < length]
        boundaries = starts + [starts[-1] + 15]
        for frame in boundaries:
            if frame >= length:
                raise ValueError("Boundary extends outside raw episode.")
            ids.append(prefix + str(frame))
            per_task[task] += 1
    ids.sort()
    if (len(ids) != len(set(ids)) or len(episode_keys) != 270
            or sorted(episode_keys) != manifest["validation_keys"]
            or set(per_task) != set(tasks)):
        raise ValueError("Validation episode/decision coverage differs from raw frozen split.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(ids, stream, indent=2)
        stream.write("\n")
    audit = {"schema":"zeva-behavior-effect-validation5-expected-v1", "split":"validation",
             "formal_labels_used":False, "episodes":len(episode_keys), "decisions":len(ids),
             "per_task_decisions":dict(sorted(per_task.items())),
             "sha256":{"adapter":digest(adapter), "task_config":digest(args.tasks),
                       "stage1_manifest":digest(args.stage1_manifest),
                       "expected_decisions":digest(args.output), "enumerator":digest(Path(__file__))}}
    with args.audit_output.open("x") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"episodes":audit["episodes"], "decisions":audit["decisions"],
                      "expected_sha256":audit["sha256"]["expected_decisions"]}), flush=True)


if __name__ == "__main__":
    main()
