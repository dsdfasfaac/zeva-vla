"""Synthetic validation-only selector contract tests (no rollout labels)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.robotwin_eval.select_h15_route_candidate import EXPECTED_ADAPTER_SHA
from scripts.robotwin_eval.select_h15_route_candidate import EXPECTED_BASE_SHA
from scripts.robotwin_eval.select_h15_route_candidate import EXPECTED_SAMPLE_SHA
from scripts.robotwin_eval.select_h15_route_candidate import select


class H15RouteCandidateSelectorTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"train_args": {
            "zeva_h15_flow_objective": True,
            "decouple_action_expert_gradient": True,
            "action_expert_learning_rate": 0.0,
        }}))
        self.manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

    def tearDown(self):
        self.tempdir.cleanup()

    def _report(self, step: int, on: float, off: float, base: float = 0.01) -> Path:
        checkpoint = self.root / f"{step:06d}"
        report = {
            "protocol": {
                "complete": True,
                "batch_size": 16,
                "seed": 1000,
                "validation_split": "validation5",
                "validation_decision_samples": 5874,
                "available_validation_batches": 368,
                "evaluated_batches": 368,
                "ordered_validation_samples_sha256": EXPECTED_SAMPLE_SHA,
                "policy_horizon": 50,
                "executed_horizon": 15,
                "optimizer_created": False,
                "checkpoint_written": False,
                "training_variant": "zeva",
                "video_backend": "torchcodec",
                "dataset_adapter_sha256": EXPECTED_ADAPTER_SHA,
                "validation_only_intervention": {
                    "context_gate_scale": 1.0,
                    "prior_gate_scale": 1.0,
                    "deployment_or_checkpoint_modified": False,
                },
            },
            "checkpoint": {
                "path": str(checkpoint),
                "model_sha256": EXPECTED_BASE_SHA,
                "adapter": {"path": str(checkpoint / "zeva_adapter.pth"), "sha256": "adapter-sha"},
            },
            "fixed_teacher": {
                "available": True,
                "model_sha256": EXPECTED_BASE_SHA,
                "shared_frozen_weights_verified": True,
            },
            "lineage": {
                "stage1_schema": "zeva-robotwin-zte-stage1-v2-checkpoint",
                "transition_horizon": 15,
                "causal_bank_sha256": "33b11b41992d07890a798602cb8cadebc0f798b8959adfc3b8cf5b660bf14ad9",
            },
            "source_manifest": {"path": str(self.root / "manifest.json"), "sha256": self.manifest_sha},
            "result": {"validation_diagnostics": {
                "flow": {
                    "zeva_residual_on_executed_h15": {
                        "available": True, "valid_examples": 5874, "mean": on,
                    },
                    "current_residual_off_executed_h15": {
                        "available": True, "valid_examples": 5874, "mean": off,
                    },
                },
                "paired": {"fixed_teacher": {
                    "available": True, "valid_examples": 5874,
                    "student_flow": on, "teacher_flow": base,
                }},
            }},
        }
        path = self.root / f"{step:06d}.json"
        path.write_text(json.dumps(report))
        return path

    def test_earliest_passing_step_is_selected(self):
        reports = {500: self._report(500, .0098, .0100), 1000: self._report(1000, .0097, .0100)}
        result = select(reports)
        self.assertEqual(result["decision"], "selected")
        self.assertEqual(result["selected_step"], 500)
        self.assertFalse(result["test_success_labels_used"])

    def test_no_passing_step_is_rejected(self):
        reports = {500: self._report(500, .0101, .0100), 1000: self._report(1000, .0102, .0100)}
        result = select(reports)
        self.assertEqual(result["decision"], "rejected_offline")
        self.assertIsNone(result["selected_checkpoint"])

    def test_changed_sample_order_is_rejected(self):
        reports = {500: self._report(500, .0098, .0100), 1000: self._report(1000, .0097, .0100)}
        path = reports[1000]
        report = json.loads(path.read_text())
        report["protocol"]["ordered_validation_samples_sha256"] = "changed"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "ordered_validation_samples_sha256"):
            select(reports)

    def test_changed_base_weights_are_rejected(self):
        reports = {500: self._report(500, .0098, .0100), 1000: self._report(1000, .0097, .0100)}
        path = reports[500]
        report = json.loads(path.read_text())
        report["checkpoint"]["model_sha256"] = "drifted"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            select(reports)


if __name__ == "__main__":
    unittest.main()
