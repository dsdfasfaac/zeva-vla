"""Unit tests for the bounded v2 evaluation staging preflight."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import prepare_robotwin_eval_ztev2 as staging


class V2EvalStagingTest(unittest.TestCase):
    def _inputs(self, root: Path) -> staging.Inputs:
        return staging.Inputs(
            handoff_root=root / "handoff",
            foundation_checkpoint=root / "best-v1",
            anchor_foundation_checkpoint=root / "best-v1",
            goal_embedding_checkpoint=root / "language",
            base_stage2_checkpoint=root / "base-step",
            zeva_stage2_checkpoint=root / "zeva-step",
            zte_checkpoint=root / "zte.pth",
            causal_bank=root / "bank.pt",
            retrieval_checkpoint=root / "retrieval.pth",
            task_manifest=None,
            model_rng_seed=staging.MODEL_RNG_SEED,
            episodes_per_task=staging.EPISODES_PER_TASK,
            absolute_start_seed=staging.ABSOLUTE_START_SEED,
        )

    def test_configs_keep_three_roles_and_explicit_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inputs = self._inputs(Path(temporary))
            configs = staging.build_configs(inputs)

            baseline = configs["robotwin_eval_ztev2_baseline.yml"]
            zeva = configs["robotwin_eval_ztev2_zeva.yml"]
            anchor = configs["robotwin_eval_ztev2_anchor.yml"]
            self.assertTrue(baseline["baseline_only"])
            self.assertEqual(baseline["stage2_checkpoint"], str(inputs.base_stage2_checkpoint))
            self.assertFalse(zeva["baseline_only"])
            for key in (
                "goal_embedding_checkpoint",
                "stage2_checkpoint",
                "zte_checkpoint",
                "causal_bank",
                "retrieval_checkpoint",
            ):
                self.assertEqual(zeva[key], str(getattr(inputs, {
                    "goal_embedding_checkpoint": "goal_embedding_checkpoint",
                    "stage2_checkpoint": "zeva_stage2_checkpoint",
                    "zte_checkpoint": "zte_checkpoint",
                    "causal_bank": "causal_bank",
                    "retrieval_checkpoint": "retrieval_checkpoint",
                }[key])))
            self.assertTrue(anchor["baseline_only"])
            self.assertNotIn("stage2_checkpoint", anchor)
            self.assertNotIn("zte_checkpoint", anchor)
            # Config output is JSON/YAML-compatible without relying on PyYAML.
            decoded = json.loads(staging._yaml_compatible_json(zeva))
            self.assertEqual(decoded, zeva)

    def test_placeholder_and_missing_paths_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(staging.PreflightError):
                staging._resolve_required("<SELECT_STAGE2_CHECKPOINT>", "stage2")
            with self.assertRaises(staging.PreflightError):
                staging._resolve_required(root / "does-not-exist", "stage2")

    def test_new_file_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yml"
            staging._write_new(path, "{}\n")
            with self.assertRaises(staging.PreflightError):
                staging._write_new(path, "{}\n")


if __name__ == "__main__":
    unittest.main()
