"""Focused CPU tests for the v2 episode sampler, collator, and targets."""

from types import SimpleNamespace

import unittest
from unittest.mock import patch
import torch

from scripts.train_robotwin_zte_v2 import Args  # noqa: E402
from scripts.train_robotwin_zte_v2 import GroupedTaskSampler  # noqa: E402
from scripts.train_robotwin_zte_v2 import PairedTaskSampler
from scripts.train_robotwin_zte_v2 import collate_robotwin_episodes  # noqa: E402
from scripts.train_robotwin_zte_v2 import compute_v2_losses  # noqa: E402
from scripts.train_robotwin_zte_v2 import _manifest
from scripts.train_robotwin_zte_v2 import _resume_metadata
from openpi.zeva.transition_encoder_v2 import TransitionEncoderV2Config


class _FakeGroupedDataset:
    indices_by_task = ((0, 2, 4), (1, 3))

    def __len__(self):
        return 5


def test_manifest_describes_effect_and_next_action_boundaries_separately():
    handoff = SimpleNamespace(root="/fixture", statistics="/fixture/stats.json")
    dataset = SimpleNamespace(task_names=("task",))
    for context in ("pre", "phase"):
        config = TransitionEncoderV2Config(action_prediction_context=context)
        with patch("scripts.train_robotwin_zte_v2._sha256", return_value="fixture"):
            manifest = _manifest(Args(action_prediction_context=context), handoff, config, dataset)
        flow = manifest["information_flow"]
        assert "jepa_and_action_prediction" not in flow
        assert "no-current-after-image" in flow["forward_effect_prediction"]
        assert ("current-after-visible" in flow["next_action_prediction"]) == (context == "phase")


def test_resume_only_migrates_documented_legacy_pre_context_default():
    legacy = {"zte_config": {"model_dim": 256}, "manifest": {"train_args": {"steps": 256}}}
    config, args = _resume_metadata(legacy)
    assert config == {"model_dim": 256, "action_prediction_context": "pre"}
    assert args == {"steps": 256, "action_prediction_context": "pre"}
    assert "action_prediction_context" not in legacy["zte_config"]
    current = {
        "zte_config": {"action_prediction_context": "phase"},
        "manifest": {"train_args": {"action_prediction_context": "phase"}},
    }
    assert _resume_metadata(current) == (current["zte_config"], current["manifest"]["train_args"])


def _episode(length: int, *, valid: bool = True):
    return {
        "images_before": torch.zeros(length, 3, 2, 6, dtype=torch.uint8),
        "images_after": torch.zeros(length, 3, 2, 6, dtype=torch.uint8),
        "actions": torch.zeros(length, 15, 16),
        "progress": torch.arange(length, dtype=torch.float32),
        "initial_progress": torch.tensor(0.0),
        "goal_embedding": torch.zeros(4),
        "task_id": torch.tensor(0),
        "episode_valid": torch.tensor(valid),
    }


def test_sampler_covers_every_episode_once_and_validation_is_deterministic():
    dataset = _FakeGroupedDataset()
    train_samplers = [
        GroupedTaskSampler(dataset, seed=17, replicas=2, rank=rank, shuffle=True)
        for rank in range(2)
    ]
    first = [list(sampler) for sampler in train_samplers]
    valid_indices = [index for values in first for index in values if index >= 0]
    assert sorted(valid_indices) == list(range(len(dataset)))
    assert len(valid_indices) == len(set(valid_indices))

    validation_samplers = [
        GroupedTaskSampler(dataset, seed=999, replicas=2, rank=rank, shuffle=False)
        for rank in range(2)
    ]
    expected = [list(sampler) for sampler in validation_samplers]
    for sampler in validation_samplers:
        sampler.set_epoch(10)
    assert [list(sampler) for sampler in validation_samplers] == expected
    valid_indices = [index for values in expected for index in values if index >= 0]
    assert sorted(valid_indices) == list(range(len(dataset)))


def test_collator_right_pads_and_marks_only_real_transitions():
    assert Args().batch_size == 8
    batch = collate_robotwin_episodes([_episode(2), _episode(4), _episode(1, valid=False)])
    assert batch["images_before"].shape == (3, 4, 3, 2, 6)
    assert batch["actions"].shape == (3, 4, 15, 16)
    assert batch["valid_mask"].tolist() == [
        [True, True, False, False],
        [True, True, True, True],
        [False, False, False, False],
    ]
    assert batch["episode_mask"].tolist() == [True, True, False]


def test_paired_sampler_keeps_cross_episode_positives_on_same_rank():
    class Dataset:
        indices_by_task = tuple(tuple(range(i * 4, i * 4 + 4)) for i in range(5))

        def __len__(self):
            return 20

    dataset = Dataset()
    all_indices = []
    for rank in range(2):
        sampler = PairedTaskSampler(dataset, batch_size=4, seed=42, replicas=2, rank=rank)
        indices = list(sampler)
        all_indices.extend(i for i in indices if i >= 0)
        for start in range(0, len(indices), 4):
            batch = [i for i in indices[start:start + 4] if i >= 0]
            for episode in batch:
                assert any(other != episode and other // 4 == episode // 4 for other in batch)
    assert sorted(all_indices) == list(range(len(dataset)))


def test_action_loss_targets_the_next_h15_and_ignores_padding():
    target_action = torch.tensor([[[[0.0]], [[3.0]], [[100.0]]]])
    predicted_action = torch.tensor([[[[3.0]], [[0.0]], [[-9.0]]]])
    valid_mask = torch.tensor([[True, True, False]])
    ones = torch.ones(1, 3, 2)
    outputs = SimpleNamespace(
        global_prompt=torch.ones(1, 2),
        task_prototypes=None,
        task_embedding=ones,
        target_action=target_action,
        predicted_action=predicted_action,
        predicted_effect=torch.zeros(1, 3, 2),
        target_effect=torch.zeros(1, 3, 2),
        causal_signal=ones,
        causal_target_signal=ones,
        phase_token=ones,
        phase_progress=torch.zeros(1, 3),
    )
    args = Args(
        effect_loss_weight=0.0,
        action_loss_weight=1.0,
        task_loss_weight=0.0,
        causal_alignment_weight=0.0,
        phase_contrastive_weight=0.0,
        phase_order_weight=0.0,
        language_consistency_weight=0.0,
    )
    losses = compute_v2_losses(
        outputs,
        outputs,
        outputs,
        torch.zeros(1, 3),
        torch.zeros(1, dtype=torch.long),
        args,
        valid_mask=valid_mask,
    )
    # t=0 predicts target action t=1 exactly; t=1 has no valid successor.
    assert abs(float(losses["action"])) < 1e-8


def test_transition_losses_ignore_right_padding():
    ones = torch.ones(1, 3, 2)
    outputs = SimpleNamespace(
        global_prompt=torch.ones(1, 2),
        task_prototypes=None,
        task_embedding=ones,
        target_action=torch.zeros(1, 3, 1, 1),
        predicted_action=torch.zeros(1, 3, 1, 1),
        predicted_effect=torch.zeros(1, 3, 2),
        target_effect=torch.tensor([[[0.0, 0.0], [0.0, 0.0], [99.0, 99.0]]]),
        causal_signal=ones,
        causal_target_signal=ones,
        phase_token=ones,
        phase_progress=torch.zeros(1, 3),
    )
    args = Args(
        effect_loss_weight=1.0,
        action_loss_weight=0.0,
        task_loss_weight=0.0,
        causal_alignment_weight=0.0,
        phase_contrastive_weight=0.0,
        phase_order_weight=0.0,
        language_consistency_weight=0.0,
    )
    losses = compute_v2_losses(
        outputs,
        outputs,
        outputs,
        torch.zeros(1, 3),
        torch.zeros(1, dtype=torch.long),
        args,
        valid_mask=torch.tensor([[True, True, False]]),
    )
    assert abs(float(losses["effect"])) < 1e-8


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )


if __name__ == "__main__":
    unittest.main()
