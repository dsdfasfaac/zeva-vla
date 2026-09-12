"""Small no-GPU tests for the bounded v2 serving smoke helpers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # Workstation-only unit tests need no ML runtime.
    np = None

import smoke_robotwin_ztev2_serving as smoke


class ServingSmokeHelperTest(unittest.TestCase):
    @unittest.skipIf(np is None, "NumPy is available in the handoff runtime, not this workstation")
    def test_synthetic_fixture_is_explicit_and_deterministic(self) -> None:
        first = smoke._synthetic_observation("scan_object", 32, 0.0)
        second = smoke._synthetic_observation("scan_object", 32, 0.0)
        self.assertEqual(first["task"], "scan_object")
        np.testing.assert_array_equal(
            first["observation"]["head_camera"]["rgb"],
            second["observation"]["head_camera"]["rgb"],
        )
        self.assertEqual(first["joint_action"]["vector"].shape, (14,))

    @unittest.skipIf(np is None, "NumPy is available in the handoff runtime, not this workstation")
    def test_action_summary_records_h50_and_h15_contract(self) -> None:
        actions = np.zeros((50, 16), dtype=np.float32)
        converted = np.zeros((50, 16), dtype=np.float32)
        summary = smoke._finite_action_summary(actions, converted)
        self.assertEqual(summary["action_shape"], [50, 16])
        self.assertEqual(summary["executed_h15_shape"], [15, 16])
        self.assertTrue(summary["finite"])
        self.assertTrue(summary["controller_finite"])

    def test_cache_summary_distinguishes_missing_state(self) -> None:
        missing = smoke._cache_summary(object())
        self.assertFalse(missing["state_present"])
        self.assertIsNone(missing["transition_count"])

    def test_report_writer_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            smoke._write_exclusive(path, {"passed": True})
            self.assertEqual(json.loads(path.read_text()), {"passed": True})
            with self.assertRaises(FileExistsError):
                smoke._write_exclusive(path, {"passed": False})


if __name__ == "__main__":
    unittest.main()
