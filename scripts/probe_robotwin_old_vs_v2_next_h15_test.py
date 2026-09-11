"""Unit tests for the comparable old-ZTE/v2 frozen phase probe."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from scripts.probe_robotwin_old_vs_v2_next_h15 import assert_disjoint_selection
from scripts.probe_robotwin_old_vs_v2_next_h15 import episode_block_bootstrap_ci
from scripts.probe_robotwin_old_vs_v2_next_h15 import expected_decision_frames
from scripts.probe_robotwin_old_vs_v2_next_h15 import fit_and_evaluate
from scripts.probe_robotwin_old_vs_v2_next_h15 import fit_fixed_ridge
from scripts.probe_robotwin_old_vs_v2_next_h15 import phase_rows
from scripts.probe_robotwin_old_vs_v2_next_h15 import select_episode_records
from scripts.probe_robotwin_old_vs_v2_next_h15 import validate_cache_payload


def _record(task: str, episode: int, length: int = 76) -> dict:
    return {"key": ("Clean", task), "episode_index": episode, "length": length}


class ProbeHelpersTest(unittest.TestCase):
    def test_selection_is_deterministic_complete_and_disjoint(self) -> None:
        tasks = ("a", "b")
        train_records = [_record(task, index) for task in tasks for index in range(8)]
        val_records = [_record(task, 100 + index) for task in tasks for index in range(5)]
        first = select_episode_records(
            train_records, task_names=tasks, count_per_task=4, seed=1000, split_offset=0
        )
        second = select_episode_records(
            train_records, task_names=tasks, count_per_task=4, seed=1000, split_offset=0
        )
        validation = select_episode_records(
            val_records, task_names=tasks, count_per_task=2, seed=1000, split_offset=1_000_003
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(validation), 4)
        self.assertEqual({row["task_name"] for row in first}, set(tasks))
        assert_disjoint_selection(first, validation)
        self.assertTrue(all(row["decision_frames"] == list(expected_decision_frames(76)) for row in first))

    def test_selection_rejects_short_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot select"):
            select_episode_records(
                [_record("a", 0, length=15)],
                task_names=("a",),
                count_per_task=1,
                seed=1000,
                split_offset=0,
            )

    def test_cache_alignment_checks_sha_identity_and_frames(self) -> None:
        record = _record("a", 0, length=46)
        expected = list(expected_decision_frames(46))
        good = {
            "schema": "zeva-robotwin-live-queries-h15-v1",
            "transition_horizon": 15,
            "task_names": ("a",),
            "zte_checkpoint_sha256": "ckpt",
            "statistics_sha256": "stats",
            "splits": {
                "train": [
                    {
                        "record_index": 0,
                        "task_id": 0,
                        "decision_frames": torch.tensor(expected),
                        "phase_queries": torch.zeros(len(expected), 128),
                    }
                ]
            },
        }
        validate_cache_payload(
            good,
            checkpoint_sha256="ckpt",
            statistics_sha256="stats",
            records_by_split={"train": [record]},
            task_names=("a",),
        )
        bad = copy.deepcopy(good)
        bad["splits"]["train"][0]["decision_frames"][1] = 99
        with self.assertRaisesRegex(ValueError, "frame mismatch"):
            validate_cache_payload(
                bad,
                checkpoint_sha256="ckpt",
                statistics_sha256="stats",
                records_by_split={"train": [record]},
                task_names=("a",),
            )
        bad_sha = copy.deepcopy(good)
        bad_sha["zte_checkpoint_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
            validate_cache_payload(
                bad_sha,
                checkpoint_sha256="ckpt",
                statistics_sha256="stats",
                records_by_split={"train": [record]},
                task_names=("a",),
            )

    def test_phase_rows_shift_next_action_and_separate_initial(self) -> None:
        phase = torch.arange(5 * 128, dtype=torch.float32).reshape(5, 128)
        actions = torch.arange(4 * 15 * 16, dtype=torch.float32).reshape(4, 15, 16)
        rows = phase_rows(phase, actions, (0, 15, 30, 45, 60), "episode", 3, 61)
        self.assertEqual(rows["phase"].shape, (3, 128))
        self.assertEqual(rows["target"].shape, (3, 240))
        self.assertTrue(np.array_equal(rows["phase"], phase[1:-1].numpy()))
        self.assertTrue(np.array_equal(rows["target"], actions[1:].reshape(3, -1).numpy()))
        self.assertTrue(np.array_equal(rows["initial_phase"], phase[:1].numpy()))
        self.assertTrue(np.array_equal(rows["initial_target"], actions[:1].reshape(1, -1).numpy()))

    def test_fixed_ridge_uses_train_only_standardization(self) -> None:
        train_x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        train_y = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        model = fit_fixed_ridge(train_x, train_y, alpha=1.0)
        prediction = model.predict(np.asarray([[4.0]]))
        changed_validation = fit_fixed_ridge(train_x, train_y, alpha=1.0)
        self.assertTrue(np.allclose(prediction, changed_validation.predict(np.asarray([[4.0]]))))
        self.assertEqual(model.alpha, 1.0)
        self.assertEqual(fit_fixed_ridge(train_x, train_y, alpha=2.0).alpha, 2.0)
        with self.assertRaises(ValueError):
            fit_fixed_ridge(train_x, train_y, alpha=-1.0)

    def test_episode_bootstrap_is_deterministic_and_blocked(self) -> None:
        errors = np.asarray([1.0, 3.0, 10.0, 12.0])
        episodes = ("a", "a", "b", "b")
        first = episode_block_bootstrap_ci(errors, episodes, replicates=50, seed=7)
        second = episode_block_bootstrap_ci(errors, episodes, replicates=50, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["episodes"], 2)
        self.assertEqual(first["mean"], 6.5)

    def test_fit_report_has_baseline_progress_and_initial_probe(self) -> None:
        rng = np.random.default_rng(5)
        train = {
            "phase": rng.normal(size=(8, 128)),
            "target": rng.normal(size=(8, 240)),
            "progress": rng.normal(size=(8, 1)),
            "episode": np.asarray(["tr0"] * 4 + ["tr1"] * 4, dtype=object),
            "task": np.asarray([0] * 4 + [1] * 4),
            "initial_phase": rng.normal(size=(2, 128)),
            "initial_target": rng.normal(size=(2, 240)),
            "initial_episode": np.asarray(["tr0", "tr1"], dtype=object),
            "initial_task": np.asarray([0, 1]),
        }
        validation = {key: value.copy() for key, value in train.items()}
        validation["episode"] = np.asarray(["va0"] * 4 + ["va1"] * 4, dtype=object)
        validation["initial_episode"] = np.asarray(["va0", "va1"], dtype=object)
        result = fit_and_evaluate(
            train,
            validation,
            alpha=1.0,
            bootstrap_replicates=10,
            bootstrap_seed=11,
            task_names=("task0", "task1"),
        )
        self.assertIn("next_h15", result)
        self.assertIn("task_mean_baseline_next_h15", result)
        self.assertIn("progress", result)
        self.assertIn("initial_phase_to_action0", result)
        self.assertEqual(result["next_h15"]["episode_block_bootstrap"]["episodes"], 2)


if __name__ == "__main__":
    unittest.main()
