import math

import pytest
import torch

from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_DIM
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_HORIZON
from openpi.zeva.robotwin_policy import RobotWinActionPrior
from openpi.zeva.robotwin_policy import RobotWinGaussianActionPrior
from openpi.zeva.robotwin_policy import RobotWinZevaPolicy
from openpi.zeva.robotwin_policy import gaussian_action_prior_nll


def test_gaussian_action_prior_shapes_and_bounds():
    model = RobotWinActionPrior(task_dim=8, phase_dim=4, context_dim=6, hidden_dim=16)
    prior = model(torch.randn(3, 8), torch.randn(3, 4), torch.randn(3, 6))

    expected = (3, ROBOTWIN_ACTION_HORIZON, ROBOTWIN_ACTION_DIM)
    assert prior.mean.shape == expected
    assert prior.log_std.shape == expected
    assert torch.all(prior.log_std >= -5.0)
    assert torch.all(prior.log_std <= 2.0)


def test_gaussian_action_prior_nll_matches_standard_normal_and_has_gradients():
    mean = torch.zeros(2, 5, ROBOTWIN_ACTION_DIM, requires_grad=True)
    log_std = torch.zeros_like(mean, requires_grad=True)
    target = torch.zeros_like(mean)
    prior = RobotWinGaussianActionPrior(mean=mean, log_std=log_std)

    loss = gaussian_action_prior_nll(prior, target)
    expected = ROBOTWIN_ACTION_DIM * 0.5 * math.log(2.0 * math.pi)
    torch.testing.assert_close(loss, loss.new_tensor(expected))

    loss.backward()
    assert mean.grad is not None
    assert log_std.grad is not None


def test_gaussian_action_prior_nll_rejects_shape_mismatch():
    prior = RobotWinGaussianActionPrior(
        mean=torch.zeros(1, 2, ROBOTWIN_ACTION_DIM),
        log_std=torch.zeros(1, 2, ROBOTWIN_ACTION_DIM),
    )
    target = torch.zeros(1, 3, ROBOTWIN_ACTION_DIM)

    with pytest.raises(ValueError, match="identical shapes"):
        gaussian_action_prior_nll(prior, target)


def test_frozen_foundation_checkpointing_keeps_real_layers_in_eval_mode():
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma_with_expert = torch.nn.Module()
            self.paligemma_with_expert.paligemma = torch.nn.Sequential(
                torch.nn.Dropout(0.5), torch.nn.Linear(2, 2)
            )
            self.paligemma_with_expert.gemma_expert = torch.nn.Module()
            self.paligemma_with_expert.gemma_expert.model = torch.nn.Sequential(
                torch.nn.Dropout(0.5), torch.nn.Linear(2, 2)
            )
            self.gradient_checkpointing_enabled = False

        def gradient_checkpointing_enable(self):
            self.gradient_checkpointing_enabled = True

    class Foundation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Core()

    policy = RobotWinZevaPolicy.__new__(RobotWinZevaPolicy)
    torch.nn.Module.__init__(policy)
    policy.foundation = Foundation()
    policy.causal_transition_encoder = torch.nn.Module()
    policy.causal_transition_encoder.target_vision_encoder = torch.nn.Dropout(0.5)
    policy.retrieval_head = torch.nn.Dropout(0.5)
    policy._full_pi05_finetune = False
    policy._action_expert_finetune = False
    policy.train()

    policy.enforce_frozen_foundation_checkpointing_mode()

    core = policy.foundation.model
    assert core.gradient_checkpointing_enabled
    assert core.training
    assert core.paligemma_with_expert.training
    assert not core.paligemma_with_expert.paligemma.training
    assert not core.paligemma_with_expert.gemma_expert.model.training
    assert not policy.causal_transition_encoder.training
    assert not policy.causal_transition_encoder.target_vision_encoder.training
    assert not policy.retrieval_head.training
    assert all(
        not module.training
        for module in core.paligemma_with_expert.paligemma.modules()
        if isinstance(module, torch.nn.Dropout)
    )
    assert all(
        not module.training
        for module in core.paligemma_with_expert.gemma_expert.model.modules()
        if isinstance(module, torch.nn.Dropout)
    )
