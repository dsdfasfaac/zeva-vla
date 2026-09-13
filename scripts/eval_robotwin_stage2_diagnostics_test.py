"""Focused tests for the read-only Stage 2 diagnostic entry point."""

# The import shim must run before importing the diagnostic script in the
# lightweight local environment; private helpers are the intended test seam.
# ruff: noqa: I001, SLF001

from __future__ import annotations

import pytest


pytest.importorskip("torch")
diagnostics = pytest.importorskip("scripts.eval_robotwin_stage2_diagnostics")


def test_gate_intervention_scales_branches_and_restores_after_error():
    class FakePolicy:
        def _activate_residual_gates(self):
            self._active_context_gate = 0.01
            self._active_prior_gate = 0.02

    policy = FakePolicy()
    with diagnostics._validation_gate_intervention(policy, 1.0, 1.0):
        assert "_activate_residual_gates" not in vars(policy)
        policy._activate_residual_gates()
        assert policy._active_prior_gate == 0.02
    with pytest.raises(RuntimeError, match="fixture"):
        with diagnostics._validation_gate_intervention(policy, 0.0, 50.0):
            policy._activate_residual_gates()
            assert policy._active_context_gate == 0.0
            assert policy._active_prior_gate == 1.0
            policy._activate_residual_gates()  # Replay must not compound scaling.
            assert policy._active_prior_gate == 1.0
            raise RuntimeError("fixture")
    assert "_activate_residual_gates" not in vars(policy)
    policy._activate_residual_gates()
    assert policy._active_context_gate == 0.01
    assert policy._active_prior_gate == 0.02
    with diagnostics._validation_gate_intervention(policy, 1.0, 0.0):
        policy._activate_residual_gates()
        assert policy._active_context_gate == 0.01
        assert policy._active_prior_gate == 0.0


@pytest.mark.parametrize("value", [-1.0, 101.0, float("inf"), float("nan")])
def test_gate_intervention_rejects_invalid_scale(value):
    with pytest.raises(ValueError, match="finite"):
        with diagnostics._validation_gate_intervention(object(), value, 1.0):
            pass


def test_action_only_teacher_requires_identical_frozen_tensors(tmp_path):
    from types import SimpleNamespace
    import torch
    from safetensors.torch import save_file

    foundation = torch.nn.Module()
    foundation.expert = torch.nn.Linear(2, 2, bias=False)
    foundation.vision = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
    student, teacher = tmp_path / "student", tmp_path / "teacher"
    student.mkdir()
    teacher.mkdir()
    values = foundation.state_dict()
    save_file(values, student / "model.safetensors")
    changed = {key: value.clone() for key, value in values.items()}
    changed["expert.weight"].add_(1)
    save_file(changed, teacher / "model.safetensors")
    diagnostics._verify_shared_frozen_weights(SimpleNamespace(foundation=foundation), student, teacher)
    changed["vision.weight"].add_(1)
    save_file(changed, teacher / "model.safetensors")
    with pytest.raises(ValueError, match="different frozen tensor"):
        diagnostics._verify_shared_frozen_weights(SimpleNamespace(foundation=foundation), student, teacher)


def test_evaluation_window_marks_any_explicit_cap_incomplete() -> None:
    assert diagnostics._evaluation_window(0, 7) == (7, 7, True)
    assert diagnostics._evaluation_window(2, 7) == (2, 2, False)
    assert diagnostics._evaluation_window(20, 7) == (20, 7, False)


def test_evaluation_window_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="eval_batches"):
        diagnostics._evaluation_window(-1, 7)
    with pytest.raises(ValueError, match="available"):
        diagnostics._evaluation_window(0, 0)


def test_output_path_is_never_overwritten(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / "diagnostics.json"

    assert diagnostics._ensure_new_output(output, checkpoint, manifest) == output.resolve()
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        diagnostics._ensure_new_output(output, checkpoint, manifest)


def test_legacy_flow_equivalence_requires_padding_free_equal_batches() -> None:
    result = {
        "flow": 0.125,
        "validation_diagnostics": {
            "flow": {
                "zeva_residual_on_h50": {
                    "mean": 0.125,
                    "valid_examples": 32,
                    "valid_steps": 32 * 50,
                }
            }
        },
    }
    report = diagnostics._legacy_unmasked_equivalence(result, batch_size=16)
    assert report["available"] is True
    assert report["equivalent"] is True

    padded = {
        **result,
        "validation_diagnostics": {
            "flow": {
                "zeva_residual_on_h50": {
                    "mean": 0.125,
                    "valid_examples": 32,
                    "valid_steps": 31 * 50,
                }
            }
        },
    }
    padded_report = diagnostics._legacy_unmasked_equivalence(padded, batch_size=16)
    assert padded_report["available"] is False
    assert "padding" in padded_report["reason"]
