from __future__ import annotations

import pytest
import torch

from openpi.zeva.robotwin_contract import prepare_robotwin_pi_image


def test_prepare_robotwin_pi_image_matches_train_and_eval_layouts() -> None:
    hwc = torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8)
    train_chw = hwc.permute(2, 0, 1).contiguous()

    eval_result = prepare_robotwin_pi_image(hwc, name="eval")
    train_result = prepare_robotwin_pi_image(train_chw, name="train")

    assert eval_result.shape == train_result.shape == (3, 480, 640)
    assert eval_result.dtype == train_result.dtype == torch.float32
    torch.testing.assert_close(eval_result, train_result, rtol=0, atol=0)
    assert 0.0 <= float(train_result.min()) <= float(train_result.max()) <= 1.0


def test_prepare_robotwin_pi_image_accepts_batched_float_contract() -> None:
    batch = torch.rand(2, 3, 480, 640)
    result = prepare_robotwin_pi_image(batch, name="batch")
    assert result.shape == (2, 3, 480, 640)
    assert result.dtype == torch.float32
    assert result.is_contiguous()


@pytest.mark.parametrize(
    "image",
    [
        torch.full((3, 480, 640), 255.0),
        torch.full((3, 480, 640), -0.01),
        torch.full((3, 480, 640), float("nan")),
    ],
)
def test_prepare_robotwin_pi_image_rejects_invalid_float_domain(image: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="pixels"):
        prepare_robotwin_pi_image(image, name="invalid")


def test_prepare_robotwin_pi_image_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="must be"):
        prepare_robotwin_pi_image(torch.zeros(3, 240, 320, dtype=torch.uint8), name="wrong")
