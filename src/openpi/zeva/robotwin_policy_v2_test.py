"""Bounded policy-level smoke tests for the schema-aware Stage 1 v2 path."""

from dataclasses import asdict
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from openpi.zeva.config import ZevaConfig
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_DIM
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_HORIZON
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_contract import MeanStdActionNormalizer
from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
from openpi.zeva.robotwin_policy import stage1_artifact_schema
from openpi.zeva.robotwin_policy import validate_stage1_v2_artifact_status
from openpi.zeva.stage1_checkpoint import V2_SCHEMA
from openpi.zeva.transition_encoder_v2 import CausalTransitionEncoderV2
from openpi.zeva.transition_encoder_v2 import TransitionEncoderV2Config
from openpi.zeva.transition_encoder_v2_test import TinyVision


class _LegacyEncoderStub(nn.Module):
    """Avoid requiring the CUDA-only legacy Mamba extension in this fixture."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.placeholder = nn.Parameter(torch.zeros(()))


class _FoundationModel(nn.Module):
    def embed_suffix(self, *args, **kwargs):
        del kwargs
        action = args[0] if args and torch.is_tensor(args[0]) else torch.zeros(1, 1, 8)
        return action, None, None, None


class _Foundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FoundationModel()
        self.config = SimpleNamespace(
            chunk_size=ROBOTWIN_ACTION_HORIZON,
            n_action_steps=ROBOTWIN_ACTION_HORIZON,
            output_features={"action": torch.empty(ROBOTWIN_ACTION_DIM)},
            input_features={"observation.state": torch.empty(14)},
            image_features=ROBOTWIN_CAMERA_KEYS,
        )
        self.register_buffer("base_chunk", torch.linspace(
            -0.5, 0.5, ROBOTWIN_ACTION_HORIZON * ROBOTWIN_ACTION_DIM
        ).reshape(ROBOTWIN_ACTION_HORIZON, ROBOTWIN_ACTION_DIM))

    def predict_action_chunk(self, batch):
        return self.base_chunk.to(batch[ROBOTWIN_CAMERA_KEYS[0]].device).unsqueeze(0).expand(
            batch[ROBOTWIN_CAMERA_KEYS[0]].shape[0], -1, -1
        ).clone()


def _v2_payload() -> dict:
    config = TransitionEncoderV2Config(
        model_dim=16,
        phase_dim=8,
        signal_dim=8,
        task_dim=8,
        goal_dim=12,
        num_views=3,
        num_mamba_layers=1,
        image_size=8,
        dropout=0.0,
        vision_pretrained=False,
        use_mamba=False,
        task_count=2,
        vision_microbatch_size=2,
    )
    model = CausalTransitionEncoderV2(config)
    return {
        "schema": V2_SCHEMA,
        "step": 512,
        "zte_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "manifest": {
            "causal_transition_horizon": 15,
            "contract": {"executed_horizon": 15, "policy_horizon": 50},
        },
    }


def _batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    return {
        key: torch.rand(batch_size, 3, 8, 8)
        for key in ROBOTWIN_CAMERA_KEYS
    } | {
        "observation.language.tokens": torch.ones(batch_size, 4, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "zeva.goal_embedding": torch.randn(batch_size, 12),
    }


class RobotWinPolicyV2Test(unittest.TestCase):
    def test_artifact_schema_requires_explicit_v2_lineage(self):
        self.assertEqual(stage1_artifact_schema({"stage1_checkpoint_schema": V2_SCHEMA}), V2_SCHEMA)
        self.assertIsNone(stage1_artifact_schema({"stage1_checkpoint_sha256": "same"}))

    def test_v2_artifact_status_is_explicit_and_complete(self):
        valid = {"incomplete": False, "usable_for_training": True}
        validate_stage1_v2_artifact_status(valid, artifact_name="fixture")
        with self.assertRaisesRegex(ValueError, "incomplete=false"):
            validate_stage1_v2_artifact_status(
                {"usable_for_training": True}, artifact_name="fixture"
            )
        with self.assertRaisesRegex(ValueError, "incomplete=false"):
            validate_stage1_v2_artifact_status(
                {"incomplete": True, "usable_for_training": False}, artifact_name="fixture"
            )
        with self.assertRaisesRegex(ValueError, "usable_for_training=true"):
            validate_stage1_v2_artifact_status(
                {"incomplete": False, "usable_for_training": False}, artifact_name="fixture"
            )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            validate_stage1_v2_artifact_status(
                {
                    "incomplete": False,
                    "usable_for_training": True,
                    "manifest": {"incomplete": True},
                },
                artifact_name="fixture",
            )

    def test_v2_checkpoint_loads_and_policy_keeps_base_h50_with_h15_state(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as handle:
            with patch("openpi.zeva.robotwin_policy.CausalTransitionEncoder", _LegacyEncoderStub):
                with patch("openpi.zeva.transition_encoder_v2.LightweightVisionEncoder", TinyVision):
                    payload = _v2_payload()
                    torch.save(payload, handle.name)
                    policy = RobotWinZevaPolicy(
                        _Foundation(),
                        MeanStdActionNormalizer(
                            torch.zeros(ROBOTWIN_ACTION_DIM),
                            torch.ones(ROBOTWIN_ACTION_DIM),
                            "fixture",
                        ),
                        zeva_config=ZevaConfig(
                            action_dim=16,
                            action_horizon=50,
                            model_dim=16,
                            phase_dim=8,
                            signal_dim=8,
                            task_dim=8,
                            goal_dim=12,
                            num_views=3,
                            task_count=2,
                            image_size=8,
                            vision_pretrained=False,
                            num_mamba_layers=1,
                        ),
                    )
                    policy.frozen_goal_embedding_table = torch.randn(32, 2048)
                    policy.load_zte(handle.name)

                    self.assertIsInstance(policy.causal_transition_encoder, CausalTransitionEncoderV2)
                    self.assertEqual(policy._stage1_schema, V2_SCHEMA)  # noqa: SLF001
                    self.assertEqual(policy._stage1_transition_horizon, 15)  # noqa: SLF001

                    first = _batch()
                    second = _batch()
                    first_action = policy.predict_action_chunk(first, reset_scope="episode")
                    second_action = policy.predict_action_chunk(second)

                    self.assertEqual(tuple(first_action.shape), (1, 50, 16))
                    self.assertEqual(tuple(second_action.shape), (1, 50, 16))
                    torch.testing.assert_close(first_action, second_action, rtol=0, atol=0)
                    self.assertEqual(  # noqa: SLF001
                        tuple(policy._pending_normalized_actions.shape), (1, 15, 16)
                    )
                    self.assertEqual(policy._stage1_state.transition_count, 1)  # noqa: SLF001
                    self.assertTrue(  # noqa: SLF001
                        policy._stage1_state.visual_cache is not None
                        or policy._stage1_state.before_images is not None
                    )


if __name__ == "__main__":
    unittest.main()
