import ast
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

release_root = Path("/mnt/100T/users/dingxin/VLA/fixed-anchor-pair-20260914-P8hRUO")
hook_path = release_root / "scripts/robotwin_eval/eval_policy_client.py"
source = hook_path.read_text(encoding="utf-8")
tree = ast.parse(source)
function = next(
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "_configure_sapien_renderer"
)
namespace = {"Any": object, "os": os, "re": re}
module = ast.Module(body=[function], type_ignores=[])
ast.fix_missing_locations(module)
exec(compile(module, str(hook_path), "exec"), namespace)
selected = namespace["_configure_sapien_renderer"]({})

import sapien.core as sapien_core

device = sapien_core.Device(selected)
engine = sapien_core.Engine()
renderer = sapien_core.SapienRenderer()
engine.set_renderer(renderer)
sapien_core.render.set_camera_shader_dir("rt")
sapien_core.render.set_ray_tracing_samples_per_pixel(1)
sapien_core.render.set_ray_tracing_path_depth(1)
scene = engine.create_scene(sapien_core.SceneConfig())
camera = scene.add_camera("formal-health-smoke", 640, 480, np.deg2rad(37), 0.1, 100.0)
camera.entity.set_pose(sapien_core.Pose([0.0, 0.0, 1.0]))
scene.step()
scene.update_render()
camera.take_picture()
image = camera.get_picture("Color")
finite = bool(np.isfinite(image).all())
result = {
    "passed": bool(tuple(image.shape) == (480, 640, 4) and finite),
    "pid": os.getpid(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "renderer_device": selected,
    "device_cuda_id": int(device.cuda_id),
    "device_pci": str(device.pci_string),
    "frame_shape": list(image.shape),
    "dtype": str(image.dtype),
    "finite": finite,
    "vk_icd_filenames": os.environ.get("VK_ICD_FILENAMES"),
    "vk_icd_exists": Path(os.environ.get("VK_ICD_FILENAMES", "")).is_file(),
    "hook_path": str(hook_path),
}
print(json.dumps(result, sort_keys=True), flush=True)
time.sleep(8)
