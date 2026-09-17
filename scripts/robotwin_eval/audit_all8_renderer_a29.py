#!/usr/bin/env python3
"""Prove all eight idle aigc29 GPUs can render and cleanly tear down SAPIEN.

Each child uses the formal renderer hook and isolated Vulkan loader/EGL files.
The parent observes a finite frame, the child PID on the expected physical GPU,
and exit code zero. This is still a host health check, not a RoboTwin rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


LOADER_DIR = Path("/mnt/100T/users/dingxin/VLA/runtime/zeva-vulkan-loader-a28-20260916")
ICD = Path("/etc/vulkan/icd.d/nvidia_icd.json")
RUNTIME = Path("/mnt/100T/users/dingxin/WAM/playground/Benchmark/RoboTwin")
PYTHON = RUNTIME / ".venv_robotwin/bin/python"


def _run(*command: str) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gpu_rows() -> list[dict[str, str | int]]:
    raw = _run(
        "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    rows = []
    for line in raw.splitlines():
        index, uuid, pci, memory, utilization = [item.strip() for item in line.split(",")]
        rows.append({"index": int(index), "uuid": uuid, "pci": pci,
                     "memory_mib": int(memory), "utilization_percent": int(utilization)})
    if len(rows) != 8 or [row["index"] for row in rows] != list(range(8)):
        raise RuntimeError("Expected exactly eight ordered physical GPUs")
    return rows


def _compute_rows() -> list[tuple[int, str]]:
    raw = _run("nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader")
    rows = []
    for line in raw.splitlines():
        if line.strip():
            pid, uuid, _used = [item.strip() for item in line.split(",", 2)]
            rows.append((int(pid), uuid))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split(".")[0] != "aigc29":
        raise RuntimeError("This health proof is pinned to aigc29")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    smoke = args.source_root / "scripts/robotwin_eval/smoke_renderer_placement.py"
    hook = args.source_root / "scripts/robotwin_eval/eval_policy_client.py"
    for required in (smoke, hook, PYTHON, ICD, LOADER_DIR / "libvulkan.so.1", LOADER_DIR / "libEGL.so.1"):
        if not required.is_file():
            raise FileNotFoundError(required)
    rows = _gpu_rows()
    if any(row["memory_mib"] >= 1024 or row["utilization_percent"] > 5 for row in rows):
        raise RuntimeError("At least one GPU is already busy")
    if _compute_rows():
        raise RuntimeError("Compute processes already own an evaluation GPU")
    args.output_dir.mkdir(parents=True)
    results = []
    for row in rows:
        index, uuid, pci = int(row["index"]), str(row["uuid"]), str(row["pci"])
        current = _gpu_rows()[index]
        if (current["uuid"] != uuid or current["memory_mib"] >= 1024
                or current["utilization_percent"] > 5 or _compute_rows()):
            raise RuntimeError(f"GPU{index} became busy before its smoke")
        frame_path = args.output_dir / f"gpu{index}.json"
        log_path = args.output_dir / f"gpu{index}.log"
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": uuid,
            "ZEVA_SAPIEN_RENDER_DEVICE": "cuda:0",
            "VK_ICD_FILENAMES": str(ICD),
            "LD_LIBRARY_PATH": f"{LOADER_DIR}:/usr/lib/x86_64-linux-gnu",
        })
        command = [
            str(PYTHON), str(smoke), "--hook", str(hook),
            "--expected-pci", pci, "--output", str(frame_path),
            "--explicit-cleanup", "--hold-seconds", "15",
        ]
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=RUNTIME, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            mapped = False
            deadline = time.monotonic() + 90
            while process.poll() is None and time.monotonic() < deadline:
                if frame_path.is_file():
                    mapped = (process.pid, uuid) in _compute_rows()
                    if mapped:
                        break
                time.sleep(0.5)
            if process.poll() is None:
                try:
                    code = process.wait(timeout=max(1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.terminate()  # Only the child started by this health probe.
                    code = process.wait(timeout=10)
            else:
                code = process.returncode
        if not frame_path.is_file():
            raise RuntimeError(f"GPU{index} produced no renderer frame, exit={code}")
        frame = json.loads(frame_path.read_text())
        same_pci = ":".join(frame["device_pci"].lower().split(":")[-2:]) == ":".join(pci.lower().split(":")[-2:])
        passed = (code == 0 and mapped and frame.get("frame_passed") is True
                  and frame.get("pid") == process.pid and frame.get("cuda_visible_devices") == uuid
                  and frame.get("renderer_device") == "cuda:0" and same_pci)
        results.append({
            "index": index, "uuid": uuid, "pci": pci, "pid": process.pid,
            "pid_mapping_verified": mapped, "frame_passed": frame.get("frame_passed"),
            "frame_sha256": _sha256(frame_path), "process_exit_code": code,
            "passed": passed, "frame_path": str(frame_path), "log_path": str(log_path),
        })
        print(f"GPU{index}: {'passed' if passed else 'failed'}", flush=True)
        if not passed:
            break
    proof = {
        "schema": "zeva-robotwin-a29-all8-renderer-health-v1",
        "host": "aigc29",
        "passed": len(results) == 8 and all(item["passed"] for item in results),
        "gpu_rows_before": rows,
        "probes": results,
        "vulkan_loader_sha256": _sha256(LOADER_DIR / "libvulkan.so.1"),
        "isolated_egl_sha256": _sha256(LOADER_DIR / "libEGL.so.1"),
        "vk_icd_filenames": str(ICD),
        "formal_rollout_started": False,
    }
    (args.output_dir / "health_proof.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    if not proof["passed"]:
        raise SystemExit("All-eight renderer health proof failed")
    print("All-eight renderer health proof passed; no rollout started", flush=True)


if __name__ == "__main__":
    main()
