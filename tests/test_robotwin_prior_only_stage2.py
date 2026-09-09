from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from openpi.zeva.robotwin_policy import RobotWinZevaPolicy


def _stub_policy() -> RobotWinZevaPolicy:
    policy = RobotWinZevaPolicy.__new__(RobotWinZevaPolicy)
    nn.Module.__init__(policy)
    policy.foundation = nn.Linear(2, 2)
    policy.causal_transition_encoder = nn.Linear(2, 2)
    policy.task_token_projector = nn.Linear(2, 2)
    policy.memory_context_encoder = nn.Linear(2, 2)
    policy.action_prior = nn.Linear(2, 2)
    policy.causal_action_projector = nn.Linear(2, 2)
    policy.prior_action_projector = nn.Linear(2, 2)
    policy.residual_gate_router = nn.Linear(2, 2)
    policy.context_gate_logit = nn.Parameter(torch.tensor(0.0))
    policy.prior_gate_logit = nn.Parameter(torch.tensor(0.0))
    policy.retrieval_head = None
    return policy


def test_prior_only_stage2_is_zero_context_and_fixed_half_gate() -> None:
    policy = _stub_policy()
    trainable = policy.configure_prior_only_adapter_stage2(prior_gate_probability=0.5)

    assert not any(parameter.requires_grad for parameter in policy.foundation.parameters())
    assert not any(
        parameter.requires_grad for parameter in policy.causal_transition_encoder.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in policy.causal_action_projector.parameters()
    )
    assert torch.count_nonzero(policy.causal_action_projector.weight) == 0
    assert torch.count_nonzero(policy.causal_action_projector.bias) == 0
    assert not policy.context_gate_logit.requires_grad
    assert not policy.prior_gate_logit.requires_grad
    assert math.isclose(float(torch.sigmoid(policy.prior_gate_logit)), 0.5, abs_tol=1e-7)

    expected_modules = (
        policy.task_token_projector,
        policy.memory_context_encoder,
        policy.action_prior,
        policy.prior_action_projector,
        policy.residual_gate_router,
    )
    assert all(
        all(parameter.requires_grad for parameter in module.parameters())
        for module in expected_modules
    )
    assert {id(parameter) for parameter in trainable} == {
        id(parameter) for module in expected_modules for parameter in module.parameters()
    }


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_prior_only_stage2_rejects_invalid_gate(probability: float) -> None:
    with pytest.raises(ValueError, match="prior_gate_probability"):
        _stub_policy().configure_prior_only_adapter_stage2(
            prior_gate_probability=probability
        )


def test_fresh_dual_residual_gate_initializer_sets_both_gates() -> None:
    policy = _stub_policy()

    policy.initialize_residual_gate_probability(0.1)

    assert math.isclose(float(torch.sigmoid(policy.context_gate_logit)), 0.1, abs_tol=1e-7)
    assert math.isclose(float(torch.sigmoid(policy.prior_gate_logit)), 0.1, abs_tol=1e-7)


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_fresh_dual_residual_gate_initializer_rejects_invalid_probability(
    probability: float,
) -> None:
    with pytest.raises(ValueError, match="initial residual gate probability"):
        _stub_policy().initialize_residual_gate_probability(probability)
