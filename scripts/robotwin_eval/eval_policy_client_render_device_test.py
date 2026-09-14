"""Focused tests for the opt-in SAPIEN renderer-device hook.

The complete RoboTwin client imports simulator-only modules, so these tests
extract the small helper and exercise it against a fake SAPIEN module.  No
simulator, GPU, SSH connection, or evaluation output is touched.
"""

import ast
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import types
import unittest


SCRIPT = Path(__file__).with_name("eval_policy_client.py")


class FakeDevice:
    def __init__(self, alias):
        self.alias = str(alias)

    def __str__(self):
        return self.alias


class FakeRenderer:
    def __init__(self, device=None, **kwargs):
        self.device = device
        self.kwargs = kwargs


def load_helper():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_configure_sapien_renderer"
    )
    namespace = {"Any": object, "os": os, "re": __import__("re")}
    # The production runtime is Python 3.10+, while the lightweight local
    # test interpreter may be 3.9; postpone annotations for the extracted
    # helper so the test remains simulator-independent.
    future_annotations = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations", asname=None)],
        level=0,
    )
    module = ast.Module(body=[future_annotations, function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace["_configure_sapien_renderer"]


@contextmanager
def fake_sapien_modules():
    package = types.ModuleType("sapien")
    core = types.ModuleType("sapien.core")
    package.__path__ = []
    package.Device = FakeDevice
    package.SapienRenderer = FakeRenderer
    core.Device = FakeDevice
    core.SapienRenderer = FakeRenderer
    package.core = core
    old_package = sys.modules.get("sapien")
    old_core = sys.modules.get("sapien.core")
    sys.modules["sapien"] = package
    sys.modules["sapien.core"] = core
    try:
        yield package, core
    finally:
        if old_package is None:
            sys.modules.pop("sapien", None)
        else:
            sys.modules["sapien"] = old_package
        if old_core is None:
            sys.modules.pop("sapien.core", None)
        else:
            sys.modules["sapien.core"] = old_core


class RenderDeviceHookTest(unittest.TestCase):
    def setUp(self):
        self.configure = load_helper()
        self.old_spec = os.environ.pop("ZEVA_SAPIEN_RENDER_DEVICE", None)
        self.old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    def tearDown(self):
        if self.old_spec is None:
            os.environ.pop("ZEVA_SAPIEN_RENDER_DEVICE", None)
        else:
            os.environ["ZEVA_SAPIEN_RENDER_DEVICE"] = self.old_spec
        if self.old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.old_visible

    def test_default_off_does_not_import_or_patch_sapien(self):
        with fake_sapien_modules() as (package, core):
            original = package.SapienRenderer
            self.assertIsNone(self.configure({}))
            self.assertIs(package.SapienRenderer, original)
            self.assertIs(core.SapienRenderer, original)

    def test_enabled_hook_passes_local_cuda_device_to_both_import_spellings(self):
        os.environ["ZEVA_SAPIEN_RENDER_DEVICE"] = "cuda:0"
        with fake_sapien_modules() as (package, core):
            selected = self.configure({})
            self.assertEqual(selected, "cuda:0")
            self.assertIs(package.SapienRenderer, core.SapienRenderer)
            renderer = package.SapienRenderer()
            self.assertEqual(renderer.device.alias, "cuda:0")

    def test_conflicting_config_or_renderer_device_is_rejected(self):
        os.environ["ZEVA_SAPIEN_RENDER_DEVICE"] = "cuda:0"
        with fake_sapien_modules() as (package, _):
            with self.assertRaisesRegex(ValueError, "render_device"):
                self.configure({"render_device": "cuda:1"})
            self.configure({})
            with self.assertRaisesRegex(RuntimeError, "conflicts"):
                package.SapienRenderer(device=FakeDevice("cuda:1"))

    def test_invalid_device_alias_is_rejected(self):
        os.environ["ZEVA_SAPIEN_RENDER_DEVICE"] = "cuda:gpu6"
        with self.assertRaisesRegex(ValueError, "cuda:N"):
            self.configure({})


if __name__ == "__main__":
    unittest.main()
