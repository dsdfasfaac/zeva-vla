"""Unit tests for the offline RoboTwin v2 pair checkpoint selector."""

from __future__ import annotations

import copy
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

    def _make_same_world_resume(self, root: Path, states: dict[str, dict], manifests: dict[str, dict], role="zeva"):
        """Rewrite one fixture as a validated in-place resume from step 4500."""
        run = root / role
        source_manifest = copy.deepcopy(manifests[role])
        current_manifest = copy.deepcopy(source_manifest)
        current_manifest.setdefault("train_args", {})["resume_checkpoint"] = str(
            (run / "004500").resolve()
        )
        source_manifest_path = run / "manifest.source_before_resume_4500.json"
        source_manifest_path.write_text(json.dumps(source_manifest, sort_keys=True))
        (run / "manifest.json").write_text(json.dumps(current_manifest, sort_keys=True))
        manifests[role] = current_manifest
        for state_path, state in states.items():
            if f"/{role}/" not in state_path:
                continue
            step = int(Path(state_path).parent.name)
            state["manifest"] = source_manifest if step <= 4500 else current_manifest

        source_checkpoint = run / "004500"
        provenance = {
            "schema": selector.RESUME_PROVENANCE_SCHEMA,
            "role": role,
            "source_step": 4500,
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_manifest": str(source_manifest_path.resolve()),
            "source_manifest_sha256": selector.sha256_file(source_manifest_path),
            "source_model": str((source_checkpoint / "model.safetensors").resolve()),
            "source_model_sha256": selector.sha256_file(source_checkpoint / "model.safetensors"),
            "source_optimizer_state": str((source_checkpoint / "training_state.pt").resolve()),
            "source_optimizer_state_sha256": selector.sha256_file(
                source_checkpoint / "training_state.pt"
            ),
            "source_adapter": str((source_checkpoint / "zeva_adapter.pth").resolve()),
            "source_adapter_sha256": selector.sha256_file(source_checkpoint / "zeva_adapter.pth"),
        }
        (run / selector.RESUME_PROVENANCE_FILENAME).write_text(
            json.dumps(provenance, sort_keys=True)
        )
        return source_manifest, current_manifest, provenance

    def _add_dataset_relocation(
        self,
        root: Path,
        states: dict[str, dict],
        manifests: dict[str, dict],
        *,
        role: str = "zeva",
    ):
        source_manifest, current_manifest, provenance = self._make_same_world_resume(
            root, states, manifests, role=role
        )
        source_dataset = Path(source_manifest["train_args"]["dataset_root"])
        destination_dataset = (root / "relocated-dataset").resolve()
        current_manifest["train_args"]["dataset_root"] = str(destination_dataset)
        current_manifest["dataset_adapter"] = str(destination_dataset / "adapter.json")
        (root / role / "manifest.json").write_text(
            json.dumps(current_manifest, sort_keys=True)
        )
        manifests[role] = current_manifest
        for state_path, state in states.items():
            if f"/{role}/" in state_path and int(Path(state_path).parent.name) > 4500:
                state["manifest"] = current_manifest

        components = {
            "source": {"sha256": "a" * 64, "files": 3, "bytes": 30},
            "eef-index": {"sha256": "b" * 64, "files": 2, "bytes": 20},
            "joint14-index": {"sha256": "c" * 64, "files": 2, "bytes": 20},
            "stats": {"sha256": "d" * 64, "files": 1, "bytes": 10},
        }
        semantic_adapter = {
            "action_horizon": 50,
            "dataset_root": "@source",
            "eef_cache_root": "@eef-index",
            "image_shape": [480, 640, 3],
            "joint_cache_root": "@joint14-index",
            "return_uint8": False,
            "schema": "egoscale-robotwin-lerobot-relative-eef16-v1",
            "split_seed": 1000,
            "splits": ["Clean", "Randomized"],
            "stats_path": "@stats",
            "tasks": None,
            "validation_fraction": 0.05,
            "video_backend": "torchcodec",
        }
        reports = {}
        for side, dataset_root, adapter_sha in (
            ("source", source_dataset, "e" * 64),
            ("destination", destination_dataset, "f" * 64),
        ):
            report = {
                "schema": selector.DATASET_IDENTITY_SCHEMA,
                "dataset_root": str(dataset_root),
                "adapter_sha256": adapter_sha,
                "semantic_adapter": semantic_adapter,
                "components": components,
            }
            report_path = root / f"{side}-dataset-identity.json"
            report_path.write_text(json.dumps(report, sort_keys=True))
            reports[side] = report_path
        provenance["dataset_relocation"] = {
            "source_report": str(reports["source"].resolve()),
            "source_report_sha256": selector.sha256_file(reports["source"]),
            "destination_report": str(reports["destination"].resolve()),
            "destination_report_sha256": selector.sha256_file(reports["destination"]),
        }
        (root / role / selector.RESUME_PROVENANCE_FILENAME).write_text(
            json.dumps(provenance, sort_keys=True)
        )
        return source_manifest, current_manifest, provenance, reports

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

    def test_same_world_resume_bridges_old_source_states_only_with_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states, manifests = self._fixture(root)
            _, current_manifest, _ = self._make_same_world_resume(root, states, manifests)
            audit = selector.inspect_run(
                root / "zeva",
                "zeva",
                state_loader=lambda path: states[str(path)],
            )
            self.assertEqual(audit.final_step, selector.FINAL_STEP)
            self.assertEqual(audit.resume_provenance["source_step"], 4500)
            self.assertTrue(audit.resume_provenance["same_topology"])
            self.assertEqual(
                audit.resume_provenance["allowed_manifest_differences"],
                ["train_args.resume_checkpoint"],
            )
            result = selector.select_pair(
                root / "baseline",
                root / "zeva",
                base_pid=0,
                zeva_pid=0,
                state_loader=lambda path: states[str(path)],
            )
            self.assertEqual(
                result["runs"]["zeva"]["resume_provenance"]["source_step"],
                4500,
            )
            self.assertEqual(
                result["runs"]["zeva"]["manifest_sha256"],
                selector.sha256_file(root / "zeva" / "manifest.json"),
            )
            self.assertEqual(current_manifest["world_size"], 4)

    def test_dataset_relocation_requires_full_identity_and_canonicalizes_pair_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states, manifests = self._fixture(root)
            _, current_manifest, provenance, _ = self._add_dataset_relocation(
                root, states, manifests
            )
            audit = selector.inspect_run(
                root / "zeva",
                "zeva",
                state_loader=lambda path: states[str(path)],
            )
            self.assertEqual(
                audit.resume_provenance["allowed_manifest_differences"],
                ["dataset_adapter", "train_args.dataset_root", "train_args.resume_checkpoint"],
            )
            self.assertEqual(
                audit.resume_provenance["dataset_relocation"]["schema"],
                selector.DATASET_IDENTITY_SCHEMA,
            )
            result = selector.select_pair(
                root / "baseline",
                root / "zeva",
                base_pid=0,
                zeva_pid=0,
                state_loader=lambda path: states[str(path)],
            )
            self.assertTrue(result["pair_checks"]["matching_manifest.dataset_adapter"])
            self.assertTrue(result["pair_checks"]["matching_train_args.dataset_root"])
            self.assertEqual(
                result["runs"]["zeva"]["resume_provenance"]["dataset_relocation"][
                    "destination_dataset_root"
                ],
                current_manifest["train_args"]["dataset_root"],
            )
            self.assertEqual(
                provenance["dataset_relocation"]["source_report_sha256"],
                selector.sha256_file(root / "source-dataset-identity.json"),
            )

    def test_dataset_relocation_rejects_changed_component_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states, manifests = self._fixture(root)
            self._add_dataset_relocation(root, states, manifests)
            destination_report = root / "destination-dataset-identity.json"
            report = json.loads(destination_report.read_text())
            report["components"]["source"]["sha256"] = "0" * 64
            destination_report.write_text(json.dumps(report, sort_keys=True))
            provenance_path = root / "zeva" / selector.RESUME_PROVENANCE_FILENAME
            provenance = json.loads(provenance_path.read_text())
            provenance["dataset_relocation"]["destination_report_sha256"] = selector.sha256_file(
                destination_report
            )
            provenance_path.write_text(json.dumps(provenance, sort_keys=True))
            with self.assertRaisesRegex(selector.SelectionError, "component content identities differ"):
                selector.inspect_run(
                    root / "zeva",
                    "zeva",
                    state_loader=lambda path: states[str(path)],
                )

    def test_dataset_relocation_rejects_report_path_not_matching_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states, manifests = self._fixture(root)
            self._add_dataset_relocation(root, states, manifests)
            provenance_path = root / "zeva" / selector.RESUME_PROVENANCE_FILENAME
            provenance = json.loads(provenance_path.read_text())
            destination_report_path = root / "destination-dataset-identity.json"
            destination_report = json.loads(destination_report_path.read_text())
            destination_report["dataset_root"] = str((root / "wrong-dataset").resolve())
            destination_report_path.write_text(json.dumps(destination_report, sort_keys=True))
            provenance["dataset_relocation"]["destination_report_sha256"] = selector.sha256_file(
                destination_report_path
            )
            provenance_path.write_text(json.dumps(provenance, sort_keys=True))
            with self.assertRaisesRegex(selector.SelectionError, "does not match the corresponding manifest"):
                selector.inspect_run(
                    root / "zeva",
                    "zeva",
                    state_loader=lambda path: states[str(path)],
                )

    def test_resume_rejects_old_and_new_manifest_swap(self):
        for step in (500, 4500, 5000):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                states, manifests = self._fixture(root)
                source, current, _ = self._make_same_world_resume(root, states, manifests)
                path = str((root / "zeva" / f"{step:06d}" / "training_state.pt").resolve())
                states[path]["manifest"] = current if step <= 4500 else source
                with self.assertRaisesRegex(selector.SelectionError, "manifest differs"):
                    selector.inspect_run(
                        root / "zeva", "zeva", state_loader=lambda path: states[str(path)]
                    )

    def test_resume_rejects_scientific_manifest_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states, manifests = self._fixture(root)
            self._make_same_world_resume(root, states, manifests)
            current = json.loads((root / "zeva" / "manifest.json").read_text())
            current["train_args"]["prior_loss_weight"] = 0.02
            (root / "zeva" / "manifest.json").write_text(json.dumps(current, sort_keys=True))
            with self.assertRaisesRegex(selector.SelectionError, "outside allowed metadata"):
                selector.inspect_run(
                    root / "zeva",
                    "zeva",
                    state_loader=lambda path: states[str(path)],
                )

    def test_resume_rejects_topology_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states, manifests = self._fixture(root)
            self._make_same_world_resume(root, states, manifests)
            current = json.loads((root / "zeva" / "manifest.json").read_text())
            current["world_size"] = 1
            current["train_args"]["gradient_accumulation_steps"] = 16
            (root / "zeva" / "manifest.json").write_text(json.dumps(current, sort_keys=True))
            with self.assertRaisesRegex(selector.SelectionError, "outside allowed metadata"):
                selector.inspect_run(
                    root / "zeva",
                    "zeva",
                    state_loader=lambda path: states[str(path)],
                )

    def test_resume_requires_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states, manifests = self._fixture(root)
            self._make_same_world_resume(root, states, manifests)
            provenance_path = root / "zeva" / selector.RESUME_PROVENANCE_FILENAME
            provenance = json.loads(provenance_path.read_text())
            del provenance["source_optimizer_state_sha256"]
            provenance_path.write_text(json.dumps(provenance, sort_keys=True))
            with self.assertRaisesRegex(selector.SelectionError, "source_optimizer_state_sha256"):
                selector.inspect_run(
                    root / "zeva",
                    "zeva",
                    state_loader=lambda path: states[str(path)],
                )

    def test_resume_requires_saved_manifest_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states, manifests = self._fixture(root)
            run = root / "zeva"
            current = copy.deepcopy(manifests["zeva"])
            current["train_args"]["resume_checkpoint"] = str((run / "004500").resolve())
            (run / "manifest.json").write_text(json.dumps(current, sort_keys=True))
            with self.assertRaisesRegex(selector.SelectionError, "resume_provenance"):
                selector.inspect_run(
                    run,
                    "zeva",
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
