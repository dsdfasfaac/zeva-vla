"""Stage1 schema, architecture and execution-contract regression tests."""

import copy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch

from openpi.zeva.config import ZevaConfig
from openpi.zeva.stage1_checkpoint import (
    V2_SCHEMA, load_stage1_encoder, stage1_policy_config, stage1_transition_horizon,
)
from openpi.zeva.transition_encoder_v2 import CausalTransitionEncoderV2, TransitionEncoderV2Config
from openpi.zeva.transition_encoder_v2_test import TinyVision


class Stage1CheckpointTest(unittest.TestCase):
    def setUp(self):
        config = TransitionEncoderV2Config(
            model_dim=16, phase_dim=8, signal_dim=8, task_dim=8, goal_dim=12,
            num_mamba_layers=1, image_size=8, dropout=0.0,
            vision_pretrained=False, use_mamba=False, task_count=3,
            action_prediction_context="phase",
        )
        self.vision_patch = patch("openpi.zeva.transition_encoder_v2.LightweightVisionEncoder", TinyVision)
        self.vision_patch.start()
        self.addCleanup(self.vision_patch.stop)
        model = CausalTransitionEncoderV2(config)
        self.payload = {
            "schema": V2_SCHEMA,
            "zte_config": {**asdict(config), "vision_pretrained": True},
            "model_state_dict": model.state_dict(),
            "manifest": {"contract": {"executed_horizon": 15, "policy_horizon": 50}},
        }

    def test_v2_roundtrip_preserves_phase_architecture_and_declared_metadata(self):
        model = load_stage1_encoder(self.payload)
        self.assertIsInstance(model, CausalTransitionEncoderV2)
        self.assertEqual(model.config.action_prediction_context, "phase")
        self.assertFalse(model.config.vision_pretrained)
        self.assertTrue(model.stage1_declared_config["vision_pretrained"])
        self.assertTrue(self.payload["zte_config"]["vision_pretrained"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, self.payload["model_state_dict"][name], rtol=0, atol=0)

    def test_shared_policy_config_keeps_dimensions_without_v2_only_fields(self):
        config = stage1_policy_config(self.payload)
        self.assertIsInstance(config, ZevaConfig)
        self.assertEqual((config.phase_dim, config.num_views, config.action_horizon), (8, 3, 50))
        self.assertFalse(config.vision_pretrained)
        self.assertEqual(stage1_transition_horizon(self.payload), 15)

    def test_conflicting_horizons_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["manifest"]["contract"]["executed_horizon"] = 10
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            load_stage1_encoder(payload)
        payload["manifest"]["contract"] = {"policy_horizon": 10}
        with self.assertRaisesRegex(ValueError, "policy horizon"):
            stage1_transition_horizon(payload)

    def test_unknown_schema_and_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            stage1_policy_config({**self.payload, "schema": "unknown"})
        payload = copy.deepcopy(self.payload)
        payload["zte_config"]["made_up_architecture_flag"] = True
        with self.assertRaises(TypeError):
            stage1_policy_config(payload)

    def test_missing_weight_rejected(self):
        payload = copy.deepcopy(self.payload)
        payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))
        with self.assertRaises(RuntimeError):
            load_stage1_encoder(payload)

    def test_legacy_config_and_explicit_horizon(self):
        payload = {"schema": "zeva-robotwin-zte-stage1-checkpoint-v5",
                   "zte_config": asdict(ZevaConfig()),
                   "manifest": {"causal_transition_horizon": 15}}
        self.assertEqual(stage1_transition_horizon(payload), 15)
        self.assertEqual(stage1_policy_config(payload).phase_dim, 128)
        payload["manifest"] = {}
        with self.assertRaisesRegex(ValueError, "explicit"):
            stage1_transition_horizon(payload)


if __name__ == "__main__":
    unittest.main()
