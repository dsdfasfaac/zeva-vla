# ruff: noqa: PT009
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PIMTrainingSettingsTest(unittest.TestCase):
    def test_registry_has_two_explicit_settings_and_valid_entrypoints(self):
        payload = json.loads((ROOT / "configs/robotwin_pim_training_settings.json").read_text())
        self.assertEqual(payload["default_setting"], "within-episode")
        self.assertEqual(set(payload["settings"]), {"cross-attempt", "within-episode"})
        for setting in payload["settings"].values():
            self.assertTrue((ROOT / setting["entrypoint"]).is_file())
            self.assertTrue((ROOT / setting["trainer"]).is_file())
            self.assertTrue((ROOT / setting["config"]).is_file())
            if "artifact_builder" in setting:
                self.assertTrue((ROOT / setting["artifact_builder"]).is_file())

    def test_describe_uses_zeva_setting_names(self):
        result = subprocess.run(
            [str(ROOT / "scripts/run_robotwin_pim_training.sh"), "--describe"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("cross-attempt", result.stdout)
        self.assertIn("within-episode", result.stdout)
        self.assertNotIn("BehaviorVLA", result.stdout)

    def test_public_setting_surfaces_use_only_zeva_names(self):
        public_paths = (
            "README.md",
            "configs/robotwin_pim_training_settings.json",
            "docs/ZEVA_PIM_TRAINING_SETTINGS.md",
            "scripts/run_robotwin_pim_training.sh",
            "scripts/run_robotwin_cross_attempt_pim_stage2.sh",
            "scripts/robotwin_eval/launch_pim_formal_condition.sh",
            "scripts/robotwin_eval/base1000_dev.yml",
            "scripts/robotwin_eval/cte_bit_eap_parent_step5000_dev.yml",
        )
        for relative_path in public_paths:
            text = (ROOT / relative_path).read_text().lower()
            self.assertNotIn("behaviorvla", text, relative_path)
            self.assertNotIn("behavior_effect", text, relative_path)
            self.assertNotIn("behavior-effect", text, relative_path)


if __name__ == "__main__":
    unittest.main()
