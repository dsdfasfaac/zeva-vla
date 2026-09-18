"""Content identity for the frozen ten tasks, without requiring other 40 videos."""
import argparse
import json
from pathlib import Path

from scripts.audit_robotwin_dataset_identity import component, sha256


def audit(root, task_subset):
    adapter = json.loads((root/"adapter.json").read_text())
    tasks = json.loads(task_subset.read_text())["task_names"]
    semantic = dict(adapter)
    components = {}
    source_tasks = {}
    for field, name in (("dataset_root", "source"), ("eef_cache_root", "eef-index"),
                        ("joint_cache_root", "joint14-index"), ("stats_path", "stats")):
        path = Path(adapter[field])
        if not path.is_absolute():
            path = root/path
        if name == "source":
            for split in adapter["splits"]:
                for task in tasks:
                    key = f"{split}/{task}"
                    source_tasks[key] = component(path/split/task)
                    print(key, source_tasks[key], flush=True)
        else:
            components[name] = component(path)
        semantic[field] = "@" + name
    return {"schema":"zeva-behavior-effect-task-data-identity-v1", "tasks":tasks,
            "dataset_root":str(root), "adapter_sha256":sha256(root/"adapter.json"),
            "components":components, "source_tasks":source_tasks, "semantic_adapter":semantic}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.dataset_root, args.task_subset)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
