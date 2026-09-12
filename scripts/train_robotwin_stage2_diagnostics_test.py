"""CPU checks for Stage 2's opt-in validation diagnostics."""

# The import shim must run before importing the trainer in the lightweight
# encoder-only environment; E402 is intentional for this focused test.
# ruff: noqa: E402

from dataclasses import dataclass
from importlib.machinery import ModuleSpec
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn


def _install_accelerate_import_shim() -> None:
    """Keep pure Stage 2 helper tests runnable without Accelerate installed."""
    try:
        import accelerate  # noqa: PLC0415
    except ModuleNotFoundError:
        accelerate = types.ModuleType("accelerate")

        class Accelerator:
            pass

        accelerate.Accelerator = Accelerator
        utils = types.ModuleType("accelerate.utils")

        class DistributedDataParallelKwargs:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        utils.DistributedDataParallelKwargs = DistributedDataParallelKwargs
        accelerate.utils = utils
        accelerate.__spec__ = ModuleSpec("accelerate", loader=None)
        utils.__spec__ = ModuleSpec("accelerate.utils", loader=None)
        sys.modules["accelerate"] = accelerate
        sys.modules["accelerate.utils"] = utils


_install_accelerate_import_shim()

from openpi.zeva.robotwin_policy import RobotWinGaussianActionPrior
from scripts.train_robotwin_stage2 import Args
from scripts.train_robotwin_stage2 import _diagnostic_action_valid_mask
from scripts.train_robotwin_stage2 import _finalize_validation_diagnostics
from scripts.train_robotwin_stage2 import _losses
from scripts.train_robotwin_stage2 import evaluate


def test_diagnostics_reject_unsynchronized_multirank_collection():
    with pytest.raises(ValueError, match="single process"):
        evaluate(
            nn.Identity(), [], None, None, None,
            SimpleNamespace(validation_diagnostics=True),
            SimpleNamespace(num_processes=2),
        )
from scripts.train_robotwin_stage2 import _masked_action_flow_per_sample
from scripts.train_robotwin_stage2 import _relative_residual_values


def test_h15_flow_uses_only_valid_prefix_and_excludes_padding():
    raw = torch.ones((2, 50, 2))
    raw[0, 15:] = 1000.0  # H50 tail must not enter H15.
    raw[1, :15] = 2.0
    raw[1, 4:7] = 900.0  # Explicitly padded prefix steps must be ignored.
    processed = {
        "action_is_pad": torch.zeros((2, 50), dtype=torch.bool),
    }
    processed["action_is_pad"][1, 4:7] = True
    valid, source = _diagnostic_action_valid_mask(
        processed,
        batch_size=2,
        horizon=50,
        device=raw.device,
    )
    assert source == "action_is_pad"
    values, steps = _masked_action_flow_per_sample(
        raw,
        horizon=15,
        valid_mask=valid[:, :15],
    )
    torch.testing.assert_close(values, torch.tensor([1.0, 2.0]))
    assert steps.tolist() == [15, 12]


def test_zero_actions_are_not_silently_treated_as_padding():
    raw = torch.zeros((1, 50, 1))
    valid, source = _diagnostic_action_valid_mask(
        {"action": raw},
        batch_size=1,
        horizon=50,
        device=raw.device,
    )
    assert source == "implicit_all_action_steps_valid"
    assert bool(valid.all())


def test_residual_amplitude_is_measured_against_embedding_not_a_gate_proxy():
    reference = torch.ones((1, 15, 4))
    context = torch.full((1, 1, 4), 2.0)
    prior = torch.full((1, 15, 4), 3.0)
    valid = torch.ones((1, 15), dtype=torch.bool)
    values = _relative_residual_values(
        reference,
        context,
        prior,
        valid_mask=valid,
        horizon=15,
    )
    # The context is explicitly broadcast over H15, so its ratio is 2 and
    # the prior ratio is 3.  No scalar gate is an input to this calculation.
    torch.testing.assert_close(values["context_relative_norm"], torch.tensor([2.0]))
    torch.testing.assert_close(values["prior_relative_norm"], torch.tensor([3.0]))


def test_final_report_keeps_current_and_fixed_teachers_separate():
    values = {
        "student_h50": [torch.tensor([1.0, 1.0])],
        "student_h15": [torch.tensor([1.0, 1.0])],
        "current_h50": [torch.tensor([2.0, 1.0])],
        "current_h15": [torch.tensor([2.0, 1.0])],
        "fixed_h50": [torch.tensor([3.0, 1.0])],
        "fixed_h15": [torch.tensor([3.0, 1.0])],
        "valid_steps_h50": [torch.tensor([50, 49])],
        "valid_steps_h15": [torch.tensor([15, 14])],
    }
    report = _finalize_validation_diagnostics(
        values,
        reasons=[],
        mask_sources=["implicit_all_action_steps_valid"],
    )
    assert report["paired"]["current_residual_off"]["label"] == "current_residual_off"
    assert report["paired"]["fixed_teacher"]["label"] == "fixed_teacher"
    assert report["paired"]["current_residual_off"]["teacher_flow"] == 1.5
    assert report["paired"]["fixed_teacher"]["teacher_flow"] == 2.0


@dataclass
class _FakePolicy(nn.Module):
    """Small policy double for checking the opt-in loss plumbing."""

    def __post_init__(self):
        super().__init__()
        self.memory_context_encoder = nn.Identity()

    def set_foundation_rng_state(self, *_args):
        raise AssertionError("diagnostic test must not need RNG replay")

    def forward(self, _processed, **_kwargs):
        batch_size = _processed["action"].shape[0]
        flow = torch.arange(batch_size, dtype=torch.float32)
        prior = RobotWinGaussianActionPrior(
            mean=torch.zeros((batch_size, 50, 16)),
            log_std=torch.zeros((batch_size, 50, 16)),
        )
        return flow, prior

    def _task_schema(self, _processed):
        return torch.zeros((2, 3))

    def injection_gate_regularizer(self, _task_schema, _phase):
        return torch.tensor(0.25)


def _fake_bank_batch():
    return SimpleNamespace(
        phase_token=torch.zeros((2, 3)),
        brief_signals=torch.zeros((2, 1, 3)),
        retrieved_signals=torch.zeros((2, 1, 3)),
        brief_mask=torch.ones((2, 1), dtype=torch.bool),
        retrieved_mask=torch.ones((2, 1), dtype=torch.bool),
    )


def test_diagnostics_default_off_preserves_existing_loss_math_and_keys():
    policy = _FakePolicy()
    processed = {"action": torch.zeros((2, 50, 16))}
    kwargs = {
        "policy": policy,
        "processed": processed,
        "bank_batch": _fake_bank_batch(),
        "injection_confidence": torch.ones(2),
        "baseline_flow": torch.ones(2),
        "foundation_rng_state": None,
        "prior_weight": 0.01,
        "preserve_weight": 1.0,
        "gate_regularization_weight": 1e-3,
        "prior_residual_dropout_probability": 0.0,
        "training": False,
    }
    old = _losses(**kwargs)
    explicit_off = _losses(**kwargs, return_diagnostics=False)
    assert Args().validation_diagnostics is False
    assert set(old) == set(explicit_off)
    for name in old:
        torch.testing.assert_close(old[name], explicit_off[name], equal_nan=True)
