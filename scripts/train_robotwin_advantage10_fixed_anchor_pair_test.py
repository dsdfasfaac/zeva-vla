"""Contract tests for the fixed-Anchor Stage 2 launcher.

This test is intentionally independent of the H100 runtime and uses only the
standard library.  It validates the launcher/configuration contract without
starting a training process or requiring any checkpoint files locally.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "train_robotwin_advantage10_fixed_anchor_pair.sh"
CONFIG = ROOT / "configs" / "robotwin_ztev2_fixed_anchor_pair_20260914.json"
BASE_SHA256 = "2f106633403e5f2146bf7cd4f56858cbfdb856e9b2966d1c724a79ac4948c84f"


class FixedAnchorPairContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text()
        cls.config = json.loads(CONFIG.read_text())

    def test_shell_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_launcher_does_not_mutate_legacy_runtime(self) -> None:
        self.assertIn('"$python_bin" -m accelerate.commands.launch', self.source)
        self.assertNotIn('bash "$zeva_root/scripts/train_robotwin_stage2_8gpu.sh"', self.source)
        self.assertNotIn("ln -s", self.source)
        self.assertNotIn("--resume-checkpoint", self.source)
        self.assertNotIn('export TMPDIR=', self.source)
        self.assertIn('export CUDA_VISIBLE_DEVICES="$gpu_list"', self.source)

    def test_fresh_fixed_anchor_initialization_is_explicit(self) -> None:
        self.assertIn("mode=${1:-both}", self.source)
        self.assertIn("baseline|zeva|both", self.source)
        self.assertIn('--handoff-root "$handoff"', self.source)
        self.assertIn('--initial-stage2-checkpoint "$base_checkpoint"', self.source)
        self.assertIn('--anchor-stage2-checkpoint "$base_checkpoint"', self.source)
        self.assertIn('[[ -z "${RESUME_CHECKPOINT:-}" ]]', self.source)
        self.assertIn('[[ -z "${ZEVA_ADAPTER_CHECKPOINT:-}" ]]', self.source)
        self.assertIn("base_sha256=${ROBOTWIN_TRAINED_BASE_SHA256:-" + BASE_SHA256 + "}", self.source)

    def test_training_contract_is_fixed(self) -> None:
        expected_lines = (
            "steps=1000",
            "warmup_steps=100",
            "save_freq=250",
            "batch_size=16",
            "gradient_accumulation_steps=4",
            "num_processes=4",
            "eval_batches=1000000",
            "action_expert_learning_rate=5e-6",
            "zeva_learning_rate=5e-5",
            "prior_loss_weight=0.01",
            "prior_residual_dropout_probability=0.4",
            "baseline_preserve_interval=4",
            "prior_injection_horizon=50",
            "compile_mode=default",
            "seed=1000",
        )
        for line in expected_lines:
            with self.subTest(line=line):
                self.assertIn(line, self.source)
        self.assertIn("--video-backend torchcodec", self.source)
        self.assertIn("--compile-model", self.source)
        self.assertIn('"$compile_cache/torchinductor"', self.source)
        self.assertIn('"$compile_cache/triton"', self.source)

    def test_config_records_matched_branches_and_lineage(self) -> None:
        config = self.config
        self.assertEqual(config["schema"], "zeva-robotwin-stage2-fixed-anchor-pair-v1")
        self.assertTrue(config["experiment"]["not_a_resume"])
        self.assertTrue(config["experiment"]["optimizer_reset"])
        self.assertEqual(config["experiment"]["candidate_step"], 1000)
        self.assertEqual(config["provenance"]["source_model_sha256"], BASE_SHA256)
        self.assertTrue(config["provenance"]["old_zeva_adapter_must_not_be_loaded"])

        paths = config["paths"]
        branches = config["branches"]
        self.assertEqual(branches["baseline"]["training_variant"], "baseline")
        self.assertIsNone(branches["baseline"]["anchor_stage2_checkpoint"])
        self.assertEqual(branches["zeva"]["training_variant"], "zeva")
        self.assertEqual(
            branches["zeva"]["anchor_stage2_checkpoint"],
            "paths.initial_stage2_checkpoint",
        )
        self.assertEqual(
            branches["zeva"]["initial_stage2_checkpoint"],
            "paths.initial_stage2_checkpoint",
        )
        self.assertIn("004500", paths["initial_stage2_checkpoint"])

    def test_config_records_full_validation_and_protocol(self) -> None:
        training = self.config["training"]
        validation = self.config["validation"]
        protocol = self.config["deployment_protocol"]
        self.assertEqual(training["global_batch_size"], 256)
        self.assertEqual(training["prior_injection_horizon"], 50)
        self.assertEqual(training["executed_horizon"], 15)
        self.assertTrue(training["compile_model"])
        self.assertEqual(validation["expected_decisions"], 5874)
        self.assertTrue(validation["full_validation"])
        self.assertGreaterEqual(validation["eval_batches_cap"], validation["expected_decisions"])
        self.assertIn("sample-weighted H50 flow", validation["reports"])
        self.assertIn("sample-weighted executed H15 flow", validation["reports"])
        self.assertEqual(protocol["model_output"], "H50")
        self.assertEqual(protocol["execution_and_replanning"], "execute H15, then recurrent replan")


if __name__ == "__main__":
    unittest.main(verbosity=2)
