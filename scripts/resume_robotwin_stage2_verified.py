"""Explicit same-world 4500->5000 ZeVA recovery, with immutable provenance.

Does not rewrite old checkpoints. Missing RNG makes this non-bit-exact.
Dataset relocation requires full content-identity reports, not path guesses.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def write_new(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def cli_args(values):
    result = []
    for name, value in values.items():
        flag = "--" + name.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            result.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            result.extend([flag, str(value)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-data-report", type=Path, required=True)
    parser.add_argument("--destination-data-report", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29572)
    parser.add_argument("--gpus", default="0,1,3,4")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    source = args.source_root.resolve()
    sys.path.insert(0, str(source / "scripts" / "robotwin_eval"))
    import select_robotwin_ztev2_pair as selector
    old = json.loads((root / "manifest.json").read_text())
    backup = root / "manifest.source_before_resume_4500.json"
    if json.loads(backup.read_text()) != old:
        raise ValueError("Run manifest no longer matches immutable pre-resume backup")
    selector._check_manifest_variant(old, "zeva")
    if old["world_size"] != 4 or old["train_args"]["gradient_accumulation_steps"] != 4:
        raise ValueError("Only original four-rank accumulation4 recovery is supported")
    if old["train_args"]["resume_checkpoint"] is not None or (root / "005000").exists():
        raise ValueError("Already resumed or final output exists")
    for key, file in (("trainer", "scripts/train_robotwin_stage2.py"),
                      ("robotwin_policy", "src/openpi/zeva/robotwin_policy.py")):
        if selector.sha256_file(source / file) != old["source_sha256"][key]:
            raise ValueError(f"Training source changed: {key}")
    checkpoint = root / "004500"
    import torch
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False, mmap=True)
    if state["manifest"] != old or state["step"] != 4500:
        raise ValueError("Source training state does not match step4500 manifest")
    if state["scheduler_state_dict"]["last_epoch"] != 4500:
        raise ValueError("Source scheduler/global step drift")
    if sorted(group["lr"] for group in state["optimizer_state_dict"]["param_groups"]) != [5e-6, 5e-5]:
        raise ValueError("Unexpected source optimizer learning rates")
    del state
    current = copy.deepcopy(old)
    train_args = current["train_args"]
    train_args["resume_checkpoint"] = str(checkpoint)
    train_args["dataset_root"] = str(args.dataset_root.resolve())
    current["dataset_adapter"] = str(args.dataset_root.resolve() / "adapter.json")
    tree = ast.parse((source / "scripts/train_robotwin_stage2.py").read_text())
    args_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Args")
    known = {node.target.id for node in args_class.body if isinstance(node, ast.AnnAssign)}
    if set(train_args) != known:
        raise ValueError("Stored CLI schema differs from actual trainer Args")
    provenance = {"schema": selector.RESUME_PROVENANCE_SCHEMA, "role": "zeva", "source_step": 4500,
                  "source_checkpoint": str(checkpoint), "source_manifest": str(backup),
                  "source_manifest_sha256": selector.sha256_file(backup), "source_host": "aigc29",
                  "destination_host": socket.gethostname(), "rng_state_saved": False,
                  "continuation": "non_bit_exact_resume_without_saved_rng_state"}
    for label, filename in (("model", "model.safetensors"), ("optimizer_state", "training_state.pt"),
                            ("adapter", "zeva_adapter.pth")):
        path = checkpoint / filename
        provenance["source_" + label] = str(path)
        provenance["source_" + label + "_sha256"] = selector.sha256_file(path)
    provenance["dataset_relocation"] = {}
    for label, path in (("source_report", args.source_data_report),
                        ("destination_report", args.destination_data_report)):
        provenance["dataset_relocation"][label] = str(path.resolve())
        provenance["dataset_relocation"][label + "_sha256"] = selector.sha256_file(path)
    command = ["bash", str(source / "scripts/train_robotwin_stage2_8gpu.sh"), *cli_args(train_args)]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": args.gpus, "ZEVA_PROCESSES": "4",
                "ZEVA_MAIN_PROCESS_PORT": str(args.port), "ROBOTWIN_HANDOFF": old["handoff_root"],
                "ROBOTWIN_RUNTIME": old["handoff_root"] + "/runtime", "PI05_PYTHON": "/usr/bin/python3",
                "NATIVE_TRANSFORMERS_RUNTIME": "/mnt/100T/users/dingxin/VLA/runtime/transformers5-runtime-zeva-20260911",
                "ZEVA_RUNTIME_DEPS": "/mnt/100T/users/dingxin/VLA/runtime/zeva-eval-deps-py310-v1"})
    gpu_ids = [int(value) for value in args.gpus.split(",")]
    if len(set(gpu_ids)) != 4:
        raise ValueError("Exactly four distinct GPU IDs required")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    available = {int(row.split(",")[0]): [int(x) for x in row.split(",")[1:]] for row in query.splitlines()}
    if any(gpu not in available or available[gpu][0] > 1024 or available[gpu][1] != 0 for gpu in gpu_ids):
        raise ValueError("Requested GPUs are not idle; no launch")
    with socket.socket() as port_test:
        port_test.bind(("", args.port))
    if not args.launch:
        print(json.dumps({"prepared_only": True, "command": command, "provenance": provenance}, indent=2))
        return
    # Exclusive files preserve earlier recovery attempts; never silently retry.
    write_new(root / selector.RESUME_PROVENANCE_FILENAME, provenance)
    selector._load_resume_context(root, "zeva", current)
    record = root / "resume_launch_20260912.json"
    if record.exists():
        raise FileExistsError(record)
    log = root / "resume_aigc24_20260912.log"
    with log.open("x") as output:
        process = subprocess.Popen(command, cwd=source, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    write_new(record, {"pid": process.pid, "host": socket.gethostname(), "command": command,
                       "cuda_visible_devices": args.gpus, "log": str(log),
                       "continuation": provenance["continuation"], "dataset_relocation": provenance["dataset_relocation"]})
    print(json.dumps({"pid": process.pid, "log": str(log), "record": str(record)}))


if __name__ == "__main__":
    main()
