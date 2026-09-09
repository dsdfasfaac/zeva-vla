import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "make_robotwin_branch_guidance_candidate.py"
SPEC = importlib.util.spec_from_file_location("branch_guidance_candidate", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _checkpoint():
    return {
        "gate": torch.tensor(-4.0),
        "projector": {
            "weight": torch.ones(3, 2),
            "bias": torch.ones(3),
        },
    }


def test_zero_probability_exactly_disables_projector():
    checkpoint = _checkpoint()
    MODULE.configure_branch(
        checkpoint,
        probability=0.0,
        gate_key="gate",
        projector_key="projector",
    )

    assert checkpoint["gate"].item() == 0.0
    assert all(torch.count_nonzero(value).item() == 0 for value in checkpoint["projector"].values())


def test_enabled_probability_sets_absolute_sigmoid_gate_without_changing_projector():
    checkpoint = _checkpoint()
    original = {name: value.clone() for name, value in checkpoint["projector"].items()}
    MODULE.configure_branch(
        checkpoint,
        probability=0.5,
        gate_key="gate",
        projector_key="projector",
    )

    torch.testing.assert_close(torch.sigmoid(checkpoint["gate"]), torch.tensor(0.5))
    for name, value in original.items():
        torch.testing.assert_close(checkpoint["projector"][name], value)


@pytest.mark.parametrize("probability", [-0.1, 1.0, float("inf"), float("nan")])
def test_invalid_probability_is_rejected(probability):
    with pytest.raises(ValueError, match="probability"):
        MODULE.configure_branch(
            _checkpoint(),
            probability=probability,
            gate_key="gate",
            projector_key="projector",
        )
