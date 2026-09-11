"""Focused contract tests for the bounded v2 artifact exporter."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.export_robotwin_zte_v2_artifacts import HORIZON
from scripts.export_robotwin_zte_v2_artifacts import LIVE_QUERY_SCHEMA
from scripts.export_robotwin_zte_v2_artifacts import V2_STAGE1_SCHEMA
from scripts.export_robotwin_zte_v2_artifacts import _collection_mask
from scripts.export_robotwin_zte_v2_artifacts import _new_bank_accumulator
from scripts.export_robotwin_zte_v2_artifacts import _assert_output_available
from scripts.export_robotwin_zte_v2_artifacts import _shard_indices
from scripts.export_robotwin_zte_v2_artifacts import aggregate_bank
from scripts.export_robotwin_zte_v2_artifacts import expected_decision_frames
from scripts.export_robotwin_zte_v2_artifacts import finalize_bank_accumulator
from scripts.export_robotwin_zte_v2_artifacts import make_live_record
from scripts.export_robotwin_zte_v2_artifacts import merge_bank_accumulators
from scripts.export_robotwin_zte_v2_artifacts import update_bank_accumulator
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

    def test_live_record_trims_right_padding_for_unequal_episode_lengths(self) -> None:
        initial = torch.zeros(128)
        phase = torch.arange(5 * 128, dtype=torch.float32).reshape(5, 128)
        causal = torch.arange(5 * 256, dtype=torch.float32).reshape(5, 256)
        record = make_live_record(
            record_index=9,
            task_id=2,
            length=46,
            initial_phase=initial,
            phase_token=phase,
            causal_signal=causal,
        )
        self.assertEqual(tuple(record["phase_queries"].shape), (4, 128))
        self.assertEqual(tuple(record["causal_signals"].shape), (3, 256))
        self.assertTrue(torch.equal(record["phase_queries"][1].float(), phase[0]))
        self.assertTrue(torch.equal(record["causal_signals"][-1].float(), causal[2]))

    def test_streaming_bank_handles_unequal_batch_lengths_without_cat(self) -> None:
        state_a = _new_bank_accumulator(
            task_count=2, phase_bins=4, task_dim=2, phase_dim=2, signal_dim=3
        )
        state_b = _new_bank_accumulator(
            task_count=2, phase_bins=4, task_dim=2, phase_dim=2, signal_dim=3
        )
        kwargs_a = {
            "task_ids": torch.tensor([0]),
            "progress": torch.tensor([[0.2, 0.8]]),
            "phase": torch.ones(1, 2, 2),
            "initial_phase": torch.tensor([[1.0, 0.0]]),
            "causal": torch.ones(1, 2, 3),
            "global_prompt_masked": torch.tensor([[1.0, 0.0]]),
            "valid_mask": torch.tensor([[True, True]]),
            "samples_per_episode": 2,
        }
        kwargs_b = {
            "task_ids": torch.tensor([1]),
            "progress": torch.tensor([[0.1, 0.3, 0.7, 0.99]]),
            "phase": torch.full((1, 4, 2), 2.0),
            "initial_phase": torch.tensor([[0.0, 1.0]]),
            "causal": torch.full((1, 4, 3), 2.0),
            "global_prompt_masked": torch.tensor([[0.0, 1.0]]),
            "valid_mask": torch.tensor([[True, True, True, False]]),
            "samples_per_episode": 2,
        }
        update_bank_accumulator(state_a, **kwargs_a)
        update_bank_accumulator(state_a, **kwargs_b)
        update_bank_accumulator(state_b, **kwargs_a)
        update_bank_accumulator(state_b, **kwargs_b)
        result = finalize_bank_accumulator(state_a)
        merged = finalize_bank_accumulator(merge_bank_accumulators([state_b]))
        for key in result:
            self.assertTrue(torch.allclose(result[key].float(), merged[key].float()), key)
        self.assertEqual(int(result["count"].sum()), 6)  # two B0 + four selected transitions

    def test_shard_indices_are_disjoint_and_cover_source_order(self) -> None:
        source = list(range(11))
        shards = [_shard_indices(source, rank=rank, world_size=3) for rank in range(3)]
        self.assertEqual(sorted(value for shard in shards for value in shard), source)
        self.assertEqual(sum((len(shard) for shard in shards), 0), len(source))
        self.assertEqual(len({value for shard in shards for value in shard}), len(source))

    def test_output_guard_refuses_existing_final_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "train_causal_bank.pt").write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                _assert_output_available(output)


if __name__ == "__main__":
    unittest.main()
