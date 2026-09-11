"""Focused contract tests for the bounded v2 artifact exporter."""

from __future__ import annotations

import copy
import unittest

import torch

from scripts.export_robotwin_zte_v2_artifacts import HORIZON
from scripts.export_robotwin_zte_v2_artifacts import LIVE_QUERY_SCHEMA
from scripts.export_robotwin_zte_v2_artifacts import V2_STAGE1_SCHEMA
from scripts.export_robotwin_zte_v2_artifacts import _collection_mask
from scripts.export_robotwin_zte_v2_artifacts import aggregate_bank
from scripts.export_robotwin_zte_v2_artifacts import expected_decision_frames
from scripts.export_robotwin_zte_v2_artifacts import validate_live_payload
from scripts.export_robotwin_zte_v2_artifacts import validate_live_record


class RobotWinV2ArtifactContractTest(unittest.TestCase):
    def test_h15_decision_frames_are_b0_then_every_after_frame(self) -> None:
        self.assertEqual(expected_decision_frames(16), (0, 15))
        self.assertEqual(expected_decision_frames(46), (0, 15, 30, 45))
        with self.assertRaises(ValueError):
            expected_decision_frames(HORIZON)

    def test_collection_mask_is_valid_unique_and_deterministic(self) -> None:
        progress = torch.tensor([[0.1, 0.2, 0.7, 0.9], [0.0, 0.4, 0.8, 0.99]])
        valid = torch.tensor([[True, True, True, False], [True, False, True, True]])
        first = _collection_mask(progress, valid, phase_bins=8, samples_per_episode=4)
        second = _collection_mask(progress, valid, phase_bins=8, samples_per_episode=4)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(~first | valid))
        self.assertLessEqual(int(first.sum()), 4 * progress.shape[0])
        self.assertTrue(torch.all(first.sum(dim=1) == torch.tensor([3, 3])))

    def test_bank_uses_real_b0_phase_and_masks_padded_transitions(self) -> None:
        task_ids = torch.tensor([0, 1])
        progress = torch.tensor([[0.2, 0.8, 0.0], [0.1, 0.9, 0.95]])
        valid = torch.tensor([[True, True, False], [True, True, True]])
        phase = torch.zeros(2, 3, 2)
        phase[0, 0] = torch.tensor([1.0, 0.0])
        phase[0, 1] = torch.tensor([0.0, 1.0])
        phase[1, 0] = torch.tensor([0.0, 1.0])
        phase[1, 1] = torch.tensor([1.0, 0.0])
        phase[1, 2] = torch.tensor([9.0, 9.0])  # right-padding-like value, must be ignored
        initial_phase = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
        causal = torch.ones(2, 3, 3)
        global_prompt = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        bank = aggregate_bank(
            task_count=3,
            phase_bins=4,
            task_dim=2,
            phase_dim=2,
            signal_dim=3,
            task_ids=task_ids,
            progress=progress,
            phase=phase,
            initial_phase=initial_phase,
            causal=causal,
            global_prompt_masked=global_prompt,
            valid_mask=valid,
            samples_per_episode=1,
        )
        self.assertEqual(int(bank["count"][0, 0]), 1)
        self.assertEqual(int(bank["count"][1].sum()), 2)
        self.assertTrue(torch.allclose(bank["phase_key"][0, 0], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.all(bank["count"][2] == 0))
        self.assertEqual(tuple(bank["task_prototype"].shape), (3, 2))

    def test_live_record_and_payload_require_stage1_provenance(self) -> None:
        frames = expected_decision_frames(46)
        record = {
            "record_index": 3,
            "task_id": 1,
            "decision_frames": torch.tensor(frames),
            "phase_queries": torch.zeros(len(frames), 128),
            "causal_signals": torch.zeros(len(frames) - 1, 256),
        }
        validate_live_record(record, length=46, task_id=1)
        payload = {
            "schema": LIVE_QUERY_SCHEMA,
            "transition_horizon": HORIZON,
            "stage1_checkpoint_schema": V2_STAGE1_SCHEMA,
            "task_names": ("a", "b"),
            "incomplete": True,
            "splits": {"train": [record], "validation": []},
        }
        validate_live_payload(payload, task_names=("a", "b"), complete=False)
        bad = copy.deepcopy(payload)
        bad["stage1_checkpoint_schema"] = "old-schema"
        with self.assertRaisesRegex(ValueError, "Stage1 checkpoint schema"):
            validate_live_payload(bad, task_names=("a", "b"), complete=False)


if __name__ == "__main__":
    unittest.main()
