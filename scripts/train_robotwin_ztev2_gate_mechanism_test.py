"""Static contract tests for the two-arm gate mechanism launcher.

These tests use only the standard library.  They parse the new contract and
shell launcher, run ``bash -n``, and never inspect remote checkpoints, invoke
Accelerate, start diagnostics, or launch training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import re
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "train_robotwin_ztev2_gate_mechanism.sh"
CONFIG = ROOT / "configs" / "robotwin_ztev2_gate_mechanism_20260915.json"
BASE_SHA256 = "2f106633403e5f2146bf7cd4f56858cbfdb856e9b2966d1c724a79ac4948c84f"
FOUNDATION_SHA256 = "7d3e945c1d17eae24b9f374d818ee43415e6a789da5587397403ea26a91e0abe"


class GateMechanismContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_shell_syntax_only(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tmpdir_resource_sharing_probe_accepts_short_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zeva-tmpdir-", dir="/tmp") as tmpdir:
            result = self._run_tmpdir_resource_sharing_probe(tmpdir)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tmpdir_resource_sharing_probe_rejects_overlong_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="zeva-tmpdir-", dir="/tmp") as parent:
            overlong = Path(parent) / ("x" * 100)
            overlong.mkdir()
            result = self._run_tmpdir_resource_sharing_probe(str(overlong))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TMPDIR resource-sharing preflight failed", result.stderr)
        self.assertIn("AF_UNIX path too long", result.stderr)

    def _run_tmpdir_resource_sharing_probe(self, tmpdir: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["TMPDIR"] = tmpdir
        return subprocess.run(
            [sys.executable, "-c", self._tmpdir_resource_sharing_probe_source()],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )

    def _tmpdir_resource_sharing_probe_source(self) -> str:
        marker = 'if ! "$python_bin" - <<\'PY\'\n'
        probe = self.source.split(marker, 1)[1].split("\nPY\n", 1)[0]
        return probe

    def test_gpu_selectors_resolve_to_unique_physical_uuids(self) -> None:
        block = self.source.split('resolved_gpu_ids=()', 1)[1].split('torch_version=$(', 1)[0]
        block = 'resolved_gpu_ids=()' + block
        mock = '''
die() { echo "$*" >&2; exit 3; }
nvidia-smi() {
  case "$2" in
    1|GPU-11111111-1111-1111-1111-111111111111) echo GPU-11111111-1111-1111-1111-111111111111 ;;
    5) echo GPU-55555555-5555-5555-5555-555555555555 ;;
    *) echo invalid ;;
  esac
}
'''
        cases = [('1 5', True), ('GPU-11111111-1111-1111-1111-111111111111 5', True),
                 ('1 1', False), ('1 GPU-11111111-1111-1111-1111-111111111111', False),
                 ('1 99', False), ('1 invalid', False)]
        for selectors, expected in cases:
            command = mock + f'gpu_ids=({selectors})\n' + block + '\necho "$CUDA_VISIBLE_DEVICES"\n'
            result = subprocess.run(['bash', '-eu', '-c', command], capture_output=True, text=True)
            with self.subTest(selectors=selectors):
                self.assertEqual(result.returncode == 0, expected, result.stderr)
                if expected:
                    self.assertEqual(result.stdout.strip(), 'GPU-11111111-1111-1111-1111-111111111111,GPU-55555555-5555-5555-5555-555555555555')

    def test_nll_routing_is_separate_opt_in_experiment(self) -> None:
        self.assertIn('if [[ "$arm" == nll_detached || "$arm" == nll_detached_full ]]; then\n    branch_args+=(--prior-nll-detach-context)', self.source)
        self.assertIn('nll_detached) arms=(nll_detached)', self.source)
        self.assertIn('"prior_nll_detach_context": arm in ("nll_detached", "nll_detached_full")', self.source)
        self.assertIn('nll-routing-20260915}', self.source)
        config = json.loads((ROOT / 'configs/robotwin_ztev2_nll_routing_20260915.json').read_text())
        self.assertTrue(config['prior_nll_detach_context'])
        self.assertEqual(config['steps'], 100)
        self.assertEqual(config['global_batch'], 256)
        self.assertEqual(config['initial_residual_gate_probability'], 0.01)

    def test_full_routing_budget_is_explicit_and_short_trial_is_not_relabelled(self) -> None:
        config = json.loads((ROOT / 'configs/robotwin_ztev2_nll_routing_full_20260915.json').read_text())
        self.assertEqual((config['steps'], config['warmup_steps'], config['save_freq']), (1000, 100, 250))
        self.assertFalse(config['resume_short_checkpoint'])
        self.assertFalse(config['short_trial_accepted'])
        self.assertIn('nll_detached_full) arms=(nll_detached_full)', self.source)
        self.assertIn('"fixed_checkpoint_step": int(steps)', self.source)

    def test_diagnostic_uses_uncapped_full_validation_api(self) -> None:
        diagnostic = self.source.split('"$zeva_root/scripts/eval_robotwin_stage2_diagnostics.py"', 1)[1]
        self.assertIn("--eval-batches 0", diagnostic)
        self.assertNotIn('--eval-batches "$eval_batches"', diagnostic)
        self.assertEqual(self.config["validation"]["diagnostic_eval_batches"], 0)

    def test_diagnostic_completion_guard_rejects_incomplete_or_changed_protocol(self) -> None:
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", self.source, re.S)
        guard = next(block for block in blocks if "Diagnostic protocol mismatch:" in block)
        fixture = {
            "protocol": {"complete": True, "validation_decision_samples": 5874,
                         "policy_horizon": 50, "executed_horizon": 15,
                         "optimizer_created": False, "checkpoint_written": False,
                         "ordered_validation_samples_sha256": "a" * 64},
            "result": {"flow": 0.01},
            "fixed_teacher": {"shared_frozen_weights_verified": True},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostic.json"
            cases = [(None, None), ("complete", False), ("validation_decision_samples", 20),
                     ("executed_horizon", 50), ("optimizer_created", True),
                     ("ordered_validation_samples_sha256", "")]
            for field, value in cases:
                payload = json.loads(json.dumps(fixture))
                if field is not None:
                    payload["protocol"][field] = value
                path.write_text(json.dumps(payload))
                result = subprocess.run([sys.executable, "-c", guard, str(path)],
                                        capture_output=True, text=True)
                with self.subTest(field=field):
                    self.assertEqual(result.returncode == 0, field is None)

    def test_two_fresh_gate_arms_only(self) -> None:
        self.assertIn("mode=${1:-both}", self.source)
        self.assertIn("gate001|gate010|both", self.source)
        self.assertIn("arms=(gate001 gate010)", self.source)
        self.assertIn("gate001) printf '%s\\n' \"0.01\"", self.source)
        self.assertIn("gate010) printf '%s\\n' \"0.10\"", self.source)
        self.assertIn("--training-variant zeva", self.source)
        self.assertIn('--initial-stage2-checkpoint "$base_checkpoint"', self.source)
        self.assertIn('--anchor-stage2-checkpoint "$base_checkpoint"', self.source)
        self.assertIn('[[ -z "${RESUME_CHECKPOINT:-}" ]]', self.source)
        self.assertIn('[[ -z "${ZEVA_ADAPTER_CHECKPOINT:-}" ]]', self.source)
        self.assertIn("not_a_matched_newly_trained_base_comparison", self.source)
        self.assertNotIn("training_variant baseline", self.source)
        self.assertNotIn("train_robotwin_stage2_8gpu.sh", self.source)
        self.assertNotIn("ln -s", self.source)

    def test_only_initial_gate_is_arm_specific(self) -> None:
        self.assertEqual(self.config["arms"]["gate001"]["training_variant"], "zeva")
        self.assertEqual(self.config["arms"]["gate010"]["training_variant"], "zeva")
        self.assertEqual(
            self.config["arms"]["gate001"]["initial_residual_gate_probability"], 0.01
        )
        self.assertEqual(
            self.config["arms"]["gate010"]["initial_residual_gate_probability"], 0.1
        )
        for arm in ("gate001", "gate010"):
            self.assertEqual(
                self.config["arms"][arm]["initial_stage2_checkpoint"],
                "paths.initial_stage2_checkpoint",
            )
            self.assertEqual(
                self.config["arms"][arm]["anchor_stage2_checkpoint"],
                "paths.initial_stage2_checkpoint",
            )

        training = self.config["training"]
        self.assertEqual(training["training_variant"], "zeva")
        self.assertTrue(training["same_seed_and_data_order"])
        self.assertTrue(training["fresh_optimizer"])
        self.assertTrue(training["zero_initialized_dual_residual_projectors"])
        self.assertEqual(training["seed"], 1000)

    def test_training_budget_and_frozen_contract(self) -> None:
        expected_lines = (
            "steps=100",
            "warmup_steps=10",
            "save_freq=100",
            "batch_size=16",
            "gradient_accumulation_steps=4",
            "num_processes=4",
            "eval_batches=1000000",
            "action_expert_learning_rate=5e-6",
            "zeva_learning_rate=5e-5",
            "prior_loss_weight=0.01",
            "prior_residual_dropout_probability=0.4",
            "memory_dropout=0.1",
            "prior_injection_horizon=50",
            "seed=1000",
        )
        for line in expected_lines:
            with self.subTest(line=line):
                self.assertIn(line, self.source)
        self.assertIn("--video-backend torchcodec", self.source)
        self.assertIn("--compile-model", self.source)
        self.assertEqual(self.config["training"]["global_batch_size"], 256)
        self.assertEqual(self.config["training"]["batch_size_per_gpu"], 16)
        self.assertEqual(self.config["training"]["gradient_accumulation_steps"], 4)
        self.assertEqual(self.config["training"]["num_processes"], 4)
        self.assertEqual(self.config["training"]["action_expert_learning_rate"], 0.000005)
        self.assertEqual(self.config["training"]["zeva_learning_rate"], 0.00005)
        self.assertEqual(self.config["training"]["prior_residual_dropout_probability"], 0.4)
        self.assertEqual(self.config["training"]["memory_dropout"], 0.1)
        self.assertEqual(self.config["training"]["prior_injection_horizon"], 50)
        self.assertEqual(self.config["training"]["executed_horizon"], 15)
        self.assertEqual(
            self.config["frozen"],
            [
                "PI0.5 vision-language backbone",
                "Stage1 ZTE/Mamba",
                "causal bank",
                "task-language retrieval",
            ],
        )

    def test_sha_lineage_is_declared_and_checked(self) -> None:
        provenance = self.config["provenance"]
        self.assertEqual(provenance["source_model_sha256"], BASE_SHA256)
        self.assertEqual(provenance["foundation_model_sha256"], FOUNDATION_SHA256)
        self.assertTrue(provenance["old_zeva_adapter_must_not_be_loaded"])
        self.assertTrue(provenance["student_and_immutable_teacher_are_same_base004500"])
        self.assertIn("actual_foundation_sha256", self.source)
        self.assertIn("actual_base_sha256", self.source)
        self.assertIn("actual_base_manifest_sha256", self.source)
        self.assertIn("Base/004500 source manifest is not training_variant=baseline", self.source)
        self.assertIn('Path(checkpoint_path).name != "004500"', self.source)
        self.assertIn('require_file "$base_checkpoint/training_state.pt"', self.source)
        for field in (
            "actual_dataset_adapter_sha256",
            "actual_zte_sha256",
            "actual_causal_bank_sha256",
            "actual_live_queries_sha256",
            "actual_task_retrieval_sha256",
        ):
            self.assertIn(field, self.source)

    def test_runtime_and_safety_contract(self) -> None:
        runtime = self.config["runtime"]
        self.assertIn("Torch2.7.1", runtime["runtime_environment"])
        self.assertIn("NATIVE_TRANSFORMERS_RUNTIME", self.source)
        self.assertIn("MODEL_DEPENDENCY_OVERLAY", self.source)
        self.assertIn("MODEL_LD_LIBRARY_PATH", self.source)
        self.assertIn("system_runtime_symlink_mutation", self.source)
        self.assertIn('[[ ! -e "$output" ]]', self.source)
        self.assertIn('[[ ! -e "$compile_cache" ]]', self.source)
        self.assertIn('[[ ! -e "$run_root/COMPLETE" ]]', self.source)
        self.assertIn("check_idle_gpus", self.source)
        self.assertIn("check_port_free", self.source)
        self.assertIn("GATE001_MAIN_PROCESS_PORT", self.source)
        self.assertIn("GATE010_MAIN_PROCESS_PORT", self.source)

    def test_final_read_only_validation_contract(self) -> None:
        validation = self.config["validation"]
        self.assertTrue(validation["full_validation"])
        self.assertEqual(validation["split"], "validation5")
        self.assertEqual(validation["expected_decisions"], 5874)
        self.assertEqual(validation["fixed_checkpoint_step"], 100)
        self.assertIn("sample-weighted H50 flow", validation["reports"])
        self.assertIn("sample-weighted executed H15 flow", validation["reports"])
        self.assertIn("ZeVA current residual-off", validation["reports"])
        self.assertIn("ZeVA fixed-004500 anchor", validation["reports"])
        self.assertTrue(validation["diagnostics_are_read_only"])
        self.assertIn("eval_robotwin_stage2_diagnostics.py", self.source)
        self.assertIn("--fixed-teacher-checkpoint", self.source)
        self.assertIn('"larger_residual_norm_is_not_utility"', self.source)
        self.assertFalse(self.config["experiment"]["formal_success_labels_used_for_training_or_selection"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
