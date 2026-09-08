import math

import pytest
import torch

from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_DIM
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_HORIZON
from openpi.zeva.robotwin_policy import RobotWinActionPrior
from openpi.zeva.robotwin_policy import RobotWinGaussianActionPrior
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
