from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from join_validate_outcome_traces import build_dataset
    from outcome_trace import OUTCOME_LABEL_SCOPE
    from outcome_trace import OutcomeTraceError
    from outcome_trace import atomic_write_episode_trace
    from outcome_trace import bind_episode_trace
    from outcome_trace import trace_filename
    from outcome_trace import validate_episode_trace
except ImportError:  # pragma: no cover - package import path in unit tests.
    from .join_validate_outcome_traces import build_dataset
    from .outcome_trace import OUTCOME_LABEL_SCOPE
    from .outcome_trace import OutcomeTraceError
    from .outcome_trace import atomic_write_episode_trace
    from .outcome_trace import bind_episode_trace
    from .outcome_trace import trace_filename
    from .outcome_trace import validate_episode_trace


def _episode(
    *,
    condition: str,
    success: bool,
    seed: int = 1000,
    episode_index: int = 0,
    split: str = "development",
    protocol: str = "chunk-start-relative-eef16-predict-h50-execute-h15",
):
    candidate = np.arange(4 * 15 * 16, dtype=np.float32).reshape(4, 15, 16) / 100.0
    record = {
        "replan_index": 0,
        "retrieved_task": "scan_object",
        "candidate_h15_actions": candidate,
        "pairwise_distances": np.zeros((4, 4), dtype=np.float32),
        "selected_candidate": 2,
        "pi_vlm_eos_feature": np.ones(2048, dtype=np.float32),
        "previous_executed_h15": None,
    }
    return bind_episode_trace(
        task="scan_object",
        seed=seed,
        episode_index=episode_index,
        instruction="scan the object",
        condition=condition,
        policy_name="zeva_policy",
        split=split,
        protocol=protocol,
        success=success,
        steps=15,
        step_limit=50,
        replans=[record],
    )


class OutcomeTraceTests(unittest.TestCase):
    def test_episode_trace_binds_provenance_and_validates_shapes(self):
        payload = _episode(condition="baseline", success=False)
        validated = validate_episode_trace(payload)
        self.assertEqual(validated["split"], "development")
        self.assertEqual(
            validated["protocol"],
            "chunk-start-relative-eef16-predict-h50-execute-h15",
        )
        self.assertEqual(validated["outcome_label_scope"], OUTCOME_LABEL_SCOPE)
        self.assertEqual(validated["replans"][0]["task"], "scan_object")
        self.assertEqual(validated["replans"][0]["split"], "development")
        self.assertEqual(validated["replans"][0]["seed"], 1000)
        self.assertEqual(
            np.asarray(validated["replans"][0]["candidate_h15_actions"]).shape,
            (4, 15, 16),
        )

    def test_episode_trace_rejects_nonfinite_and_cross_episode_identity(self):
        payload = _episode(condition="zeva", success=True)
        payload["replans"][0]["candidate_h15_actions"][0][0][0] = float("nan")
        with self.assertRaisesRegex(OutcomeTraceError, "non-finite"):
            validate_episode_trace(payload)

        payload = _episode(condition="zeva", success=True)
        payload["replans"][0]["seed"] = 1001
        with self.assertRaisesRegex(OutcomeTraceError, "does not match"):
            validate_episode_trace(payload)

        payload = _episode(condition="zeva", success=True)
        payload["outcome_label_scope"] = "per_decision_failure_causality"
        with self.assertRaisesRegex(OutcomeTraceError, "per-decision"):
            validate_episode_trace(payload)

    def test_join_requires_exact_identity_and_writes_development_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_dir = root / "baseline"
            zeva_dir = root / "zeva"
            baseline = _episode(condition="baseline", success=False)
            zeva = _episode(condition="zeva", success=True)
            atomic_write_episode_trace(
                baseline_dir / trace_filename("scan_object", 0, 1000), baseline
            )
            atomic_write_episode_trace(zeva_dir / trace_filename("scan_object", 0, 1000), zeva)
            output = root / "outcome_dataset.json"
            dataset = build_dataset(baseline_dir, zeva_dir, output=output)
            self.assertEqual(dataset["total_episodes"], 1)
            self.assertEqual(dataset["baseline_successes"], 0)
            self.assertEqual(dataset["zeva_successes"], 1)
            self.assertEqual(
                dataset["join_key"],
                ["split", "protocol", "task", "seed", "instruction"],
            )
            self.assertEqual(dataset["success_labels_supervision"], "development_only")
            self.assertEqual(
                json.loads(output.read_text())["schema"],
                "zeva-robotwin-outcome-dataset-v2",
            )

    def test_join_rejects_identity_or_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_dir = root / "baseline"
            zeva_dir = root / "zeva"
            atomic_write_episode_trace(
                baseline_dir / trace_filename("scan_object", 0, 1000),
                _episode(condition="baseline", success=False),
            )
            changed = _episode(condition="zeva", success=True, seed=1001)
            atomic_write_episode_trace(
                zeva_dir / trace_filename("scan_object", 0, 1001), changed
            )
            with self.assertRaisesRegex(OutcomeTraceError, "identity sets differ"):
                build_dataset(baseline_dir, zeva_dir, output=root / "out.json")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_dir = root / "baseline"
            zeva_dir = root / "zeva"
            atomic_write_episode_trace(
                baseline_dir / trace_filename("scan_object", 0, 1000),
                _episode(condition="baseline", success=False),
            )
            atomic_write_episode_trace(
                zeva_dir / trace_filename("scan_object", 0, 1000),
                _episode(condition="zeva", success=True, protocol="other-protocol"),
            )
            with self.assertRaisesRegex(OutcomeTraceError, "split/protocol"):
                build_dataset(baseline_dir, zeva_dir, output=root / "out.json")

    def test_join_refuses_final_test_success_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_dir = root / "baseline"
            zeva_dir = root / "zeva"
            atomic_write_episode_trace(
                baseline_dir / trace_filename("scan_object", 0, 1000),
                _episode(condition="baseline", success=False, split="test"),
            )
            atomic_write_episode_trace(
                zeva_dir / trace_filename("scan_object", 0, 1000),
                _episode(condition="zeva", success=True, split="test"),
            )
            output = root / "out.json"
            with self.assertRaisesRegex(OutcomeTraceError, "evaluation-only"):
                build_dataset(baseline_dir, zeva_dir, output=output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
