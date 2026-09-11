"""Information-flow tests, runnable on CPU or ZEVA_TEST_CUDA=1 with real Mamba."""
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from openpi.zeva.transition_encoder_v2 import CausalTransitionEncoderV2, TransitionEncoderV2Config


class TinyVision(nn.Module):
    def __init__(self, output_dim, **kwargs):
        super().__init__()
        self.project = nn.Linear(9, output_dim)

    def forward(self, images):
        return self.project(images.mean(dim=(-1, -2)).flatten(1))


class InformationFlowTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.device = "cuda" if os.environ.get("ZEVA_TEST_CUDA") == "1" else "cpu"
        config = TransitionEncoderV2Config(
            model_dim=16, phase_dim=8, signal_dim=8, task_dim=8, goal_dim=12,
            num_mamba_layers=1, image_size=8, dropout=0.0,
            vision_pretrained=False, use_mamba=self.device == "cuda", task_count=3,
        )
        with patch("openpi.zeva.transition_encoder_v2.LightweightVisionEncoder", TinyVision):
            self.model = CausalTransitionEncoderV2(config).to(self.device).eval()
        self.before = torch.rand(2, 3, 3, 8, 24, device=self.device)
        self.after = torch.rand_like(self.before)
        self.actions = torch.randn(2, 3, 15, 16, device=self.device)
        self.goal = torch.randn(2, 12, device=self.device)

    def run_model(self, before=None, actions=None, after=None, **kwargs):
        return self.model(
            self.before if before is None else before,
            self.actions if actions is None else actions,
            self.after if after is None else after,
            self.goal, **kwargs,
        )

    def test_after_target_cannot_leak_into_forward_prediction(self):
        left = self.run_model()
        right = self.run_model(after=self.after.flip(1))
        torch.testing.assert_close(left.predicted_effect, right.predicted_effect, rtol=0, atol=0)
        torch.testing.assert_close(left.predicted_action, right.predicted_action, rtol=0, atol=0)
        self.assertGreater((left.causal_signal - right.causal_signal).abs().max().item(), 1e-5)

    def test_future_cannot_change_prior_transition(self):
        left = self.run_model()
        before, after, actions = self.before.clone(), self.after.clone(), self.actions.clone()
        before[:, 1:] = 0.2
        after[:, 1:] = 0.8
        actions[:, 1:] *= -3
        right = self.run_model(before=before, after=after, actions=actions)
        for name in ("phase_token", "causal_signal", "predicted_action", "predicted_effect"):
            torch.testing.assert_close(getattr(left, name)[:, 0], getattr(right, name)[:, 0], rtol=0, atol=1e-5)

    def test_action_order_changes_representation(self):
        left = self.run_model()
        right = self.run_model(actions=self.actions.flip(2))
        self.assertGreater((left.pre_context - right.pre_context).abs().max().item(), 1e-5)

    def test_action_recurrence_does_not_cross_episode_batch_dimension(self):
        full = self.run_model()
        single = self.model(self.before[:1], self.actions[:1], self.after[:1], self.goal[:1])
        torch.testing.assert_close(full.pre_context[:1], single.pre_context, rtol=0, atol=1e-5)
        changed = self.actions.clone()
        changed[1] *= -10
        alternative = self.run_model(actions=changed)
        torch.testing.assert_close(full.pre_context[:1], alternative.pre_context[:1], rtol=0, atol=1e-5)

    def test_padding_does_not_change_valid_outputs_or_pool(self):
        mask = torch.tensor([[True, True, False], [True, True, False]], device=self.device)
        padded = self.run_model(valid_mask=mask)
        short = self.run_model(before=self.before[:, :2], after=self.after[:, :2], actions=self.actions[:, :2])
        torch.testing.assert_close(padded.global_prompt, short.global_prompt, rtol=0, atol=1e-5)
        torch.testing.assert_close(padded.phase_token[:, :2], short.phase_token, rtol=0, atol=1e-5)

    def test_streaming_reference_matches_batch_without_double_normalization(self):
        full = self.run_model()
        _, state = self.model.initialize_phase_state(self.before[:, 0], self.goal)
        for index in range(3):
            out, state = self.model.forward_step(
                self.before[:, index], self.actions[:, index], self.after[:, index], state,
            )
            torch.testing.assert_close(out.phase_token[:, -1], full.phase_token[:, index], rtol=0, atol=1e-5)

    def test_effect_prediction_has_action_gradient_but_no_after_gradient(self):
        actions = self.actions.clone().requires_grad_()
        after = self.after.clone().requires_grad_()
        output = self.run_model(actions=actions, after=after)
        output.predicted_effect.square().mean().backward()
        self.assertGreater(actions.grad.abs().sum().item(), 0)
        self.assertIsNone(after.grad)

    def test_real_vision_training_cannot_mix_future_or_padding_via_batchnorm(self):
        # Exercise actual ResNet rather than the tiny fixture for the temporal
        # leakage risk introduced by flattening episode frames into a batch.
        config = TransitionEncoderV2Config(
            model_dim=16, phase_dim=8, signal_dim=8, task_dim=8, goal_dim=12,
            num_mamba_layers=1, image_size=32, dropout=0.0,
            vision_pretrained=False, use_mamba=self.device == "cuda",
            vision_microbatch_size=2,
        )
        model = CausalTransitionEncoderV2(config).to(self.device).train()
        for module in model.vision_encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                self.assertFalse(module.training)
        first = model(self.before, self.actions, self.after, self.goal)
        changed = self.before.clone()
        changed[:, 1:] = 0.5
        second = model(changed, self.actions, self.after, self.goal)
        torch.testing.assert_close(first.predicted_effect[:, 0], second.predicted_effect[:, 0], atol=1e-5, rtol=0)
        first.predicted_effect.square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.vision_encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
