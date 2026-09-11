"""Unit tests for the offline RoboTwin v2 pair checkpoint selector."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import select_robotwin_ztev2_pair as selector
except ImportError:  # pragma: no cover - package import path in test runners.
    from . import select_robotwin_ztev2_pair as selector


def _manifest(root: Path, role: str) -> dict:
    variant = selector.EXPECTED_VARIANTS[role]
    task_subset = root / "task_subset.json"
    dataset_root = root / "dataset"
    shared = {
        "handoff_root": str(root / "handoff"),
        "foundation_checkpoint": str(root / "handoff" / "checkpoint" / "pretrained_model-best-v1"),
        "foundation_identity": {"model_sha256": "foundation", "config_sha256": "config"},
        "goal_embedding_identity": {"model_sha256": "goal", "config_sha256": "goal-config"},
        "foundation_forward": {
            "attention": "released_joint_paligemma_action_expert_forward",
            "paligemma_requires_grad": False,
            "injection_before_vlm": False,
            "torch_compile": True,
            "torch_compile_mode": "default",
        },
        "runtime_versions": {
            "accelerate": "1.13.0",
            "transformers": "5.5.4",
            "tokenizers": "0.22.2",
            "tokenizers_module": "0.21.4",
            "torch": "2.7.1+cu126",
        },
        "pi_image_contract": {
            "layout": "CHW",
            "dtype": "float32",
            "range": [0.0, 1.0],
            "source_uint8_transform": "x / 255",
            "foundation_visual_processor": "IDENTITY",
        },
        "dataset_adapter": str(dataset_root / "adapter.json"),
        "train_split": "train95",
        "validation_split": "validation5",
        "task_scope": {
            "manifest": str(task_subset),
            "manifest_sha256": "task-subset-sha",
            "mode": "specialization_subset",
            "task_names": ["task_a", "task_b"],
        },
        "effective_global_batch_size": 256,
        "world_size": 4,
        "video_decode": {"backend": "torchcodec", "persistent_decoder_cache": True},
    }
    train_args = {
        "training_variant": variant,
        "steps": 5_000,
        "warmup_steps": 500,
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "num_workers": 4,
        "video_backend": "torchcodec",
        "decoder_threads": 1,
        "action_expert_learning_rate": 5e-6,
        "learning_rate": 5e-5,
        "weight_decay": 1e-10,
        "adam_beta2": 0.95,
        "prior_loss_weight": 0.01,
        "preserve_loss_weight": 1.0,
        "paired_improvement_margin": 0.0,
        "baseline_preserve_interval": 4,
        "gate_regularization_weight": 1e-3,
        "phase_noise_std": 0.02,
        "memory_dropout": 0.1,
        "prior_residual_dropout_probability": 0.4,
        "initial_residual_gate_probability": 0.01,
        "save_freq": 500,
        "save_checkpoints": True,
        "eval_batches": 32,
        "log_freq": 10,
        "compile_model": True,
        "compile_mode": "default",
        "seed": 1000,
        "dataset_root": str(dataset_root),
        "task_subset": str(task_subset),
    }
    return {
        "schema": "zeva-robotwin-stage2-action-expert-manifest-v11",
        **shared,
        "training_variant": variant,
        "training_mode": "baseline" if role == "baseline" else "zeva",
        "causal_transition_horizon": 15,
        "causal_context_residual": {"broadcast_horizon": 50},
        "optimizer_groups": {"pi05_action_expert": {"learning_rate": 5e-6}},
        "train_args": train_args,
    }


class PairSelectorTest(unittest.TestCase):
    def _fixture(self, root: Path, *, max_step: int = selector.FINAL_STEP):
        states: dict[str, dict] = {}
        manifests: dict[str, dict] = {}
        for role in ("baseline", "zeva"):
            run = root / role
            run.mkdir()
            manifest = _manifest(root, role)
            manifests[role] = manifest
            # The selector reads the parsed manifest, while the test loader
            # supplies exact state metadata without requiring local torch.
            (run / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
            for step in range(selector.SAVE_FREQ, max_step + selector.SAVE_FREQ, selector.SAVE_FREQ):
                checkpoint = run / f"{step:06d}"
                checkpoint.mkdir()
                (checkpoint / "model.safetensors").write_bytes(f"{role}-model-{step}".encode())
                state_path = checkpoint / "training_state.pt"
                state_path.write_bytes(b"metadata")
                if role == "zeva":
                    (checkpoint / "zeva_adapter.pth").write_bytes(f"adapter-{step}".encode())
                states[str(state_path.resolve())] = {
                    "schema": selector.EXPECTED_STATE_SCHEMAS[role],
                    "step": step,
                    "manifest": manifest,
                    "validation": {
                        "flow": (
                            {500: 0.5, 1000: 0.2, 1500: 0.2}.get(step, 0.4)
                            if role == "baseline"
                            else {500: 0.9, 1000: 0.8, 1500: 0.7, 2000: 0.7}.get(step, 0.75)
                        )
                    },
                }
        return states, manifests

    def test_selects_independent_minimum_flow_and_earlier_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            states, _ = self._fixture(Path(temporary))
            result = selector.select_pair(
                Path(temporary) / "baseline",
                Path(temporary) / "zeva",
                base_pid=0,
                zeva_pid=0,
                state_loader=lambda path: states[str(path)],
            )
            self.assertEqual(result["runs"]["baseline"]["selected"]["step"], 1000)
            self.assertEqual(result["runs"]["zeva"]["selected"]["step"], 1500)
            self.assertEqual(len(result["runs"]["baseline"]["candidates"]), 10)
            self.assertEqual(len(result["runs"]["zeva"]["candidates"]), 10)
            self.assertEqual(
                result["runs"]["baseline"]["candidates"][2]["selection_reason"],
                "rejected: exact validation.flow tie lost to earlier step",
            )
            self.assertTrue(result["runs"]["zeva"]["selected"]["model_sha256"])
            self.assertTrue(result["runs"]["zeva"]["selected"]["adapter_sha256"])

    def test_final_and_all_saved_candidates_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            states, _ = self._fixture(Path(temporary), max_step=1500)
            with self.assertRaisesRegex(selector.SelectionError, "missing candidate steps"):
                selector.inspect_run(
                    Path(temporary) / "baseline",
                    "baseline",
                    state_loader=lambda path: states[str(path)],
                )

    def test_state_manifest_and_variant_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            states, manifests = self._fixture(Path(temporary))
            first_state = next(path for path in states if "/baseline/000500/" in path)
            states[first_state]["manifest"] = {**manifests["baseline"], "runtime_versions": {"bad": True}}
            with self.assertRaisesRegex(selector.SelectionError, "manifest differs"):
                selector.inspect_run(
                    Path(temporary) / "baseline",
                    "baseline",
                    state_loader=lambda path: states[str(path)],
                )

    def test_pair_runtime_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states, manifests = self._fixture(root)
            bad_manifest = {**manifests["zeva"], "runtime_versions": {"transformers": "4.53.2"}}
            (root / "zeva" / "manifest.json").write_text(json.dumps(bad_manifest, sort_keys=True))
            for path, state in states.items():
                if "/zeva/" in path:
                    state["manifest"] = bad_manifest
            with self.assertRaisesRegex(selector.SelectionError, "matched pair contract failed"):
                selector.select_pair(
                    root / "baseline",
                    root / "zeva",
                    base_pid=0,
                    zeva_pid=0,
                    state_loader=lambda path: states[str(path)],
                )

    def test_mmap_is_required_and_not_silently_downgraded(self):
        calls = []

        def fake_load(*args, **kwargs):
            calls.append((args, kwargs))
            return {"schema": "test"}

        fake_torch = types.SimpleNamespace(load=fake_load)
        path = Path("training_state.pt")
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(selector._torch_load_training_state(path)["schema"], "test")
        self.assertEqual(calls[0][1]["map_location"], "cpu")
        self.assertFalse(calls[0][1]["weights_only"])
        self.assertTrue(calls[0][1]["mmap"])

    def test_live_pid_is_rejected_and_zero_is_explicitly_stopped(self):
        with self.assertRaisesRegex(selector.SelectionError, "still running"):
            selector._pid_status(__import__("os").getpid(), "test")
        self.assertFalse(selector._pid_status(0, "test")["running"])

    def test_output_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "selected.json"
            selector._write_new(output, {"ok": True})
            with self.assertRaisesRegex(selector.SelectionError, "overwrite"):
                selector._write_new(output, {"ok": False})


if __name__ == "__main__":
    unittest.main()
