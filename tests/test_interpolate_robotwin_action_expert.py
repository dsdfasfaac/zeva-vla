from pathlib import Path

import pytest
from safetensors.torch import load_file
from safetensors.torch import save_file
import torch

from scripts.interpolate_robotwin_action_expert import interpolate


def test_interpolates_only_action_path(tmp_path: Path):
    anchor_path = tmp_path / "anchor.safetensors"
    trained_path = tmp_path / "trained.safetensors"
    output_path = tmp_path / "out" / "model.safetensors"
    save_file(
        {
            "model.action_in_proj.weight": torch.tensor([0.0, 2.0]),
            "model.paligemma_with_expert.paligemma.frozen": torch.tensor([3.0]),
        },
        anchor_path,
    )
    save_file(
        {
            "model.action_in_proj.weight": torch.tensor([2.0, 6.0]),
            "model.paligemma_with_expert.paligemma.frozen": torch.tensor([3.0]),
        },
        trained_path,
    )
    manifest = interpolate(anchor_path, trained_path, output_path, 0.25)
    output = load_file(output_path)
    assert torch.equal(output["model.action_in_proj.weight"], torch.tensor([0.5, 3.0]))
    assert torch.equal(
        output["model.paligemma_with_expert.paligemma.frozen"], torch.tensor([3.0])
    )
    assert manifest["action_tensor_count"] == 1
    assert manifest["verified_frozen_tensor_count"] == 1


def test_rejects_changed_frozen_tensor(tmp_path: Path):
    anchor_path = tmp_path / "anchor.safetensors"
    trained_path = tmp_path / "trained.safetensors"
    save_file({"model.frozen": torch.tensor([1.0])}, anchor_path)
    save_file({"model.frozen": torch.tensor([2.0])}, trained_path)
    with pytest.raises(ValueError, match="frozen tensor changed"):
        interpolate(anchor_path, trained_path, tmp_path / "out.safetensors", 0.25)
