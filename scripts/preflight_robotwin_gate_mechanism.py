"""Bounded real-weight checks for both fresh gates; never starts training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    release = Path(__file__).resolve().parents[1]
    contract = json.loads((release / "configs/robotwin_ztev2_fixed_anchor_pair_20260914.json").read_text())
    paths = contract["paths"]
    runtime = json.loads(args.runtime_manifest.read_text())
    # Imports resolve to this isolated source snapshot, not the historical release.
    pythonpath = [str(release / "src"), str(release)] + [
        item for item in runtime["pythonpath"]
        if "/fixed-anchor-pair-20260914-P8hRUO" not in item
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=":".join(pythonpath),
               LD_LIBRARY_PATH=runtime["ld_library_path"]["value"],
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               LIBRARY_PATH="/usr/local/cuda/lib64/stubs")
    for variable, directory in (("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                                ("TRITON_CACHE_DIR", "triton"),
                                ("CUDA_CACHE_PATH", "cuda"),
                                ("XDG_CACHE_HOME", "xdg"),
                                ("TORCH_HOME", "torch"),
                                ("HF_HOME", "hf")):
        env[variable] = str(args.output / directory)
    result = {"passed": False, "optimizer_created": False,
              "runtime_manifest": str(args.runtime_manifest), "release": str(release),
              "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"), "checks": []}
    for gate in (0.01, 0.10):
        label = "gate001" if gate == 0.01 else "gate010"
        report = args.output / f"{label}.json"
        command = [sys.executable, str(release / "scripts/smoke_robotwin_zeva_v2.py"),
                   "--handoff-root", paths["handoff_root"],
                   "--foundation-checkpoint", paths["foundation_checkpoint"],
                   "--initial-stage2-checkpoint", paths["initial_stage2_checkpoint"],
                   "--goal-embedding-checkpoint", paths["stage1_language_checkpoint"],
                   "--zte-checkpoint", paths["stage1_zte_checkpoint"],
                   "--causal-bank", paths["stage1_causal_bank"],
                   "--live-queries", paths["stage1_live_queries"],
                   "--retrieval-checkpoint", paths["stage1_task_retrieval"],
                   "--initial-residual-gate-probability", str(gate),
                   "--task", "pick_dual_bottles", "--output", str(report)]
        # Check compiled immutable-teacher behavior on the changed initialization.
        if gate == 0.10:
            command.append("--check-compiled-anchor")
        with (args.output / f"{label}.log").open("x") as log:
            try:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=900)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 124
        payload = json.loads(report.read_text()) if report.exists() else {}
        check = {"gate": gate, "exit_code": code, "report": str(report),
                 "passed": code == 0 and payload.get("passed") is True}
        result["checks"].append(check)
        print(json.dumps(check), flush=True)
        if not check["passed"]:
            break
    result["passed"] = len(result["checks"]) == 2 and all(x["passed"] for x in result["checks"])
    with (args.output / "completion.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
