"""Integration checks for Stage 2's schema-aware Stage 1 artifact lineage."""

from dataclasses import asdict
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import types
import unittest

import torch


def _install_accelerate_import_shim() -> None:
    """Keep manifest tests runnable in the lightweight encoder-only venv."""
    try:
        import accelerate  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        accelerate = types.ModuleType("accelerate")

        class Accelerator:  # noqa: D101 - import-only test shim.
            pass

        accelerate.Accelerator = Accelerator
        utils = types.ModuleType("accelerate.utils")

        class DistributedDataParallelKwargs:  # noqa: D101 - import-only shim.
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        utils.DistributedDataParallelKwargs = DistributedDataParallelKwargs
        accelerate.utils = utils
        accelerate.__spec__ = ModuleSpec("accelerate", loader=None)
        utils.__spec__ = ModuleSpec("accelerate.utils", loader=None)
        sys.modules["accelerate"] = accelerate
        sys.modules["accelerate.utils"] = utils


_install_accelerate_import_shim()

from openpi.zeva.config import ZevaConfig
from openpi.zeva.stage1_checkpoint import LEGACY_SCHEMAS
from openpi.zeva.stage1_checkpoint import V2_SCHEMA
from openpi.zeva.transition_encoder_v2 import TransitionEncoderV2Config
from scripts.train_robotwin_stage2 import Args
from scripts.train_robotwin_stage2 import _ConnectedRawFlowCapture
from scripts.train_robotwin_stage2 import _connected_executed_flow_per_sample
from scripts.train_robotwin_stage2 import _manifest


class ConnectedH15FlowCaptureTest(unittest.TestCase):
    def test_same_forward_h50_equivalence_and_h15_gradient(self):
        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.0))
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                return self.scale * x

        core = Core()
        capture = _ConnectedRawFlowCapture(core)
        raw_input = torch.arange(2 * 50 * 32, dtype=torch.float32).reshape(2, 50, 32) / 1000
        capture.begin()
        ordinary = core.forward(raw_input)
        connected = capture.take()
        h15, h50 = _connected_executed_flow_per_sample(connected, expected_batch_size=2)
        torch.testing.assert_close(h50, ordinary[:, :, :16].mean(dim=(1, 2)))
        torch.testing.assert_close(h15, ordinary[:, :15, :16].mean(dim=(1, 2)))
        h15.mean().backward()
        torch.testing.assert_close(core.scale.grad, raw_input[:, :15, :16].mean())
        self.assertEqual(core.calls, 1)
        self.assertIsNone(capture._raw)

    def test_capture_is_disabled_between_training_calls(self):
        core = torch.nn.Linear(2, 2)
        capture = _ConnectedRawFlowCapture(core)
        core.forward(torch.ones(1, 2))
        with self.assertRaisesRegex(RuntimeError, "did not invoke"):
            capture.take()
        capture.begin()
        with self.assertRaisesRegex(RuntimeError, r"\[B,H,D\]"):
            core.forward(torch.ones(1, 2))
        capture.abort()


class Stage2V2ManifestTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.base = root / "foundation"
        (self.base / "tokenizer").mkdir(parents=True)
        (self.base / "config.json").write_text("{}")
        (self.base / "tokenizer" / "tokenizer.json").write_text("{}")
        self.statistics = root / "statistics.json"
        self.statistics.write_text("{}")
        self.live_queries = root / "live_queries.pt"
        self.live_queries.write_bytes(b"live")
        self.task_retrieval = root / "task_retrieval.pth"
        self.task_retrieval.write_bytes(b"retrieval")
        self.causal_bank = root / "causal_bank.pt"
        self.causal_bank.write_bytes(b"bank")
        self.handoff = SimpleNamespace(
            root=root,
            checkpoint=self.base,
            statistics=self.statistics,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _checkpoint(self, schema=V2_SCHEMA):
        if schema == V2_SCHEMA:
            config = TransitionEncoderV2Config(
                model_dim=16,
                phase_dim=8,
                signal_dim=8,
                task_dim=8,
                goal_dim=12,
                num_views=3,
                task_count=2,
                image_size=8,
                use_mamba=False,
                vision_pretrained=False,
            )
        else:
            config = ZevaConfig(
                model_dim=16,
                phase_dim=8,
                signal_dim=8,
                task_dim=8,
                goal_dim=12,
                num_views=3,
                task_count=2,
                image_size=8,
                vision_pretrained=False,
            )
        return {
            "schema": schema,
            "step": 512,
            "zte_config": asdict(config),
            "manifest": {
                "causal_transition_horizon": 15,
                "contract": {"executed_horizon": 15, "policy_horizon": 50},
            },
        }

    def _args_and_checkpoint(self, schema=V2_SCHEMA):
        checkpoint_path = Path(self.tempdir.name) / f"zte-{schema}.pth"
        torch.save(self._checkpoint(schema), checkpoint_path)
        args = Args(
            zte_checkpoint=str(checkpoint_path),
            causal_bank=str(self.causal_bank),
            live_queries=str(self.live_queries),
            task_retrieval=str(self.task_retrieval),
        )
        return args

    def _bank(self, args, *, schema=V2_SCHEMA, horizon=15):
        from scripts.train_robotwin_stage2 import _sha256

        manifest = {
            "stage1_checkpoint_sha256": _sha256(args.zte_checkpoint),
            "statistics_sha256": _sha256(self.statistics),
            "stage1_checkpoint_schema": schema,
            "causal_transition_horizon": horizon,
        }
        torch.save(
            {
                "schema": "zeva-robotwin-train-causal-bank-v2",
                "split": "train95",
                "stage1_checkpoint_schema": schema,
                "incomplete": False,
                "usable_for_training": True,
                "manifest": manifest,
            },
            self.causal_bank,
        )
        return SimpleNamespace(manifest=manifest)

    def _rewrite_bank_status(self, bank, *, incomplete=None, usable_for_training=None):
        payload = torch.load(self.causal_bank, map_location="cpu", weights_only=False)
        if incomplete is not None:
            payload["incomplete"] = incomplete
        if usable_for_training is not None:
            payload["usable_for_training"] = usable_for_training
        payload["manifest"] = bank.manifest
        torch.save(payload, self.causal_bank)

    def test_v2_manifest_records_lineage_and_rejects_missing_schema(self):
        args = self._args_and_checkpoint()
        bank = self._bank(args)
        manifest = _manifest(args, self.handoff, bank, {"torch": "fixture"})
        self.assertEqual(manifest["zte_checkpoint_schema"], V2_SCHEMA)
        self.assertEqual(manifest["causal_transition_horizon"], 15)

        bank.manifest.pop("stage1_checkpoint_schema")
        self._rewrite_bank_status(bank)
        with self.assertRaisesRegex(ValueError, "explicit.*schema"):
            _manifest(args, self.handoff, bank, {})

    def test_v2_manifest_rejects_mismatched_bank_horizon_or_schema(self):
        args = self._args_and_checkpoint()
        bank = self._bank(args, horizon=14)
        with self.assertRaisesRegex(ValueError, "transition horizon"):
            _manifest(args, self.handoff, bank, {})

        bank.manifest["causal_transition_horizon"] = 15
        bank.manifest["stage1_checkpoint_schema"] = next(iter(LEGACY_SCHEMAS))
        self._rewrite_bank_status(bank)
        with self.assertRaisesRegex(ValueError, "Stage 2 v2 requires.*schema"):
            _manifest(args, self.handoff, bank, {})

    def test_v2_manifest_rejects_incomplete_or_missing_status_metadata(self):
        args = self._args_and_checkpoint()
        bank = self._bank(args)
        self._rewrite_bank_status(bank, incomplete=True, usable_for_training=False)
        with self.assertRaisesRegex(ValueError, "incomplete=false"):
            _manifest(args, self.handoff, bank, {})

        self._rewrite_bank_status(bank, incomplete=None, usable_for_training=None)
        payload = torch.load(self.causal_bank, map_location="cpu", weights_only=False)
        payload.pop("incomplete")
        payload.pop("usable_for_training")
        torch.save(payload, self.causal_bank)
        with self.assertRaisesRegex(ValueError, "incomplete=false"):
            _manifest(args, self.handoff, bank, {})

    def test_legacy_v5_manifest_stays_compatible_without_schema_field(self):
        schema = "zeva-robotwin-zte-stage1-checkpoint-v5"
        args = self._args_and_checkpoint(schema)
        bank = self._bank(args, schema=V2_SCHEMA)
        bank.manifest.pop("stage1_checkpoint_schema")
        manifest = _manifest(args, self.handoff, bank, {})
        self.assertEqual(manifest["zte_checkpoint_schema"], schema)
        self.assertEqual(manifest["causal_transition_horizon"], 15)

    def test_baseline_manifest_marks_stage1_as_lineage_only(self):
        args = self._args_and_checkpoint()
        args.training_variant = "baseline"
        bank = self._bank(args)
        manifest = _manifest(args, self.handoff, bank, {})
        self.assertEqual(manifest["stage1_usage"]["used_for_predictions"], False)
        self.assertEqual(manifest["stage1_usage"]["used_for_training"], False)
        self.assertEqual(manifest["stage1_usage"]["memory_streams"], {"brief": False, "persistent": False})
        self.assertEqual(manifest["causal_context_residual"]["direct_injection_enabled"], False)
        self.assertEqual(manifest["action_prior"]["distribution"], "disabled")


if __name__ == "__main__":
    unittest.main()
