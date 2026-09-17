#!/usr/bin/env python3
"""One-frame SAPIEN smoke through the actual RoboTwin renderer pinning hook.

This is a health test, not a RoboTwin episode or a success-rate measurement.
The caller must verify the process PID on the expected physical GPU while
this script briefly remains alive after rendering.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np


def _load_renderer_hook(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_configure_sapien_renderer"
    ]
    if len(functions) != 1:
        raise RuntimeError("Expected exactly one evaluator renderer-selection hook")
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"Any": Any, "os": os, "re": re}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_configure_sapien_renderer"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hook", required=True, type=Path)
    parser.add_argument("--expected-pci", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--explicit-cleanup", action="store_true")
    parser.add_argument("--hold-seconds", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.hold_seconds <= 60:
        raise ValueError("hold-seconds must be in [1, 60]")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite health proof: {args.output}")
    if os.environ.get("ZEVA_SAPIEN_RENDER_DEVICE") != "cuda:0":
        raise RuntimeError("Health smoke requires the formal cuda:0 SAPIEN selector")
    icd = Path(os.environ.get("VK_ICD_FILENAMES", ""))
    if not icd.is_file():
        raise RuntimeError("The formal Vulkan ICD is missing")
    selected = _load_renderer_hook(args.hook)(config={})
    import sapien.core as sapien_core  # noqa: PLC0415

    device = sapien_core.Device(selected)
    engine = sapien_core.Engine()
    renderer = sapien_core.SapienRenderer()
    engine.set_renderer(renderer)
    sapien_core.render.set_camera_shader_dir("rt")
    sapien_core.render.set_ray_tracing_samples_per_pixel(1)
    sapien_core.render.set_ray_tracing_path_depth(1)
    scene = engine.create_scene(sapien_core.SceneConfig())
    camera = scene.add_camera("zeva-placement-health", 640, 480, np.deg2rad(37), 0.1, 100.0)
    camera.entity.set_pose(sapien_core.Pose([0.0, 0.0, 1.0]))
    scene.step()
    scene.update_render()
    camera.take_picture()
    image = camera.get_picture("Color")
    # nvidia-smi prints an eight-digit PCI domain while SAPIEN often prints
    # four digits.  Compare bus:device.function after the independently
    # checked GPU UUID, not their cosmetic domain width.
    pci = ":".join(str(device.pci_string).lower().split(":")[-2:])
    expected_pci = ":".join(args.expected_pci.lower().split(":")[-2:])
    frame_passed = (
        tuple(image.shape) == (480, 640, 4)
        and bool(np.isfinite(image).all())
        and pci == expected_pci
    )
    result = {
        "schema": "zeva-robotwin-renderer-frame-probe-v1",
        "host": os.uname().nodename.split(".")[0],
        # A complete host health proof additionally needs a clean *process*
        # exit and an external PID-to-physical-GPU check.  SAPIEN can render
        # a finite frame and still raise DeviceLostError during destruction.
        "frame_passed": frame_passed,
        "process_exit_verified": False,
        "physical_pid_mapping_verified": False,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "renderer_device": selected,
        "device_cuda_id": int(device.cuda_id),
        "device_pci": str(device.pci_string),
        "expected_pci": args.expected_pci,
        "frame_shape": list(image.shape),
        "finite": bool(np.isfinite(image).all()),
        "vk_icd_filenames": str(icd),
        "hook_path": str(args.hook.resolve()),
        "success_labels_used": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    time.sleep(args.hold_seconds)
    if not frame_passed:
        raise RuntimeError("Rendered frame did not satisfy formal placement/shape/finite checks")
    if args.explicit_cleanup:
        # Distinguish a rendering failure from a C++ teardown failure.  This
        # experiment is opt-in and does not alter the formal evaluator.
        print("renderer-cleanup: camera/scene", flush=True)
        del image, camera, scene
        gc.collect()
        print("renderer-cleanup: engine/renderer/device", flush=True)
        del engine, renderer, device
        gc.collect()
        print("renderer-cleanup: complete", flush=True)


if __name__ == "__main__":
    main()
