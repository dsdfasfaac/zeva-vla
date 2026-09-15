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


def test_nll_input_detachment_preserves_values_and_prior_head_gradients():
    torch.manual_seed(34)
    model = RobotWinActionPrior(task_dim=8, phase_dim=4, context_dim=6, hidden_dim=16)
    inputs = [torch.randn(2, width, requires_grad=True) for width in (8, 4, 6)]
    ordinary = model(*inputs)
    auxiliary = model(*inputs, detach_inputs=True)
    assert torch.equal(ordinary.mean, auxiliary.mean)
    assert torch.equal(ordinary.log_std, auxiliary.log_std)
    loss = gaussian_action_prior_nll(auxiliary, torch.zeros_like(auxiliary.mean))
    gradients = torch.autograd.grad(loss, inputs + list(model.parameters()), allow_unused=True)
    assert all(value is None for value in gradients[:3])
    assert any(value is not None and torch.count_nonzero(value) for value in gradients[3:])
    flow_gradients = torch.autograd.grad(ordinary.mean.square().mean(), inputs)
    assert all(torch.count_nonzero(value) for value in flow_gradients)


def test_policy_nll_routing_keeps_both_flow_paths_attached():
    torch.manual_seed(35)
    policy = RobotWinZevaPolicy.__new__(RobotWinZevaPolicy)
    torch.nn.Module.__init__(policy)

    class Context(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(12, 6)

        def forward(self, task, phase, *unused):
            return self.linear(torch.cat([task, phase], dim=-1))

    class Foundation(torch.nn.Module):
        def forward(self, batch, reduction):
            # Exercise the actual policy.forward routing, not a real PI model.
            return policy._active_causal_context.square().mean() + policy._active_action_prior.square().mean()

    policy.memory_context_encoder = Context()
    policy.action_prior = RobotWinActionPrior(task_dim=8, phase_dim=4, context_dim=6, hidden_dim=16)
    policy.foundation = Foundation()
    policy._task_schema = lambda batch: batch["task_schema"]
    policy._activate_residual_gates = lambda *unused: None
    policy._foundation_rng_state_override = None
    task = torch.randn(2, 8, requires_grad=True)
    phase = torch.randn(2, 4)
    batch = {"task_schema": task}
    ordinary_flow, ordinary_prior = policy(batch, bank_phase_token=phase)
    flow, auxiliary = policy(batch, bank_phase_token=phase, prior_nll_detach_context=True)
    assert torch.equal(flow, ordinary_flow)
    assert torch.equal(auxiliary.mean, ordinary_prior.mean)
    shared = [task] + list(policy.memory_context_encoder.parameters())
    nll = gaussian_action_prior_nll(auxiliary, torch.zeros_like(auxiliary.mean))
    assert all(x is None for x in torch.autograd.grad(nll, shared, allow_unused=True))
    inputs = shared + list(policy.action_prior.parameters())
    gradients = torch.autograd.grad(flow, inputs, allow_unused=True)
    assert all(g is not None and torch.count_nonzero(g) for g in gradients[:len(shared)])
    assert any(g is not None and torch.count_nonzero(g) for g in gradients[len(shared):])


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
