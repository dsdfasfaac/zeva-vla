"""Real CUDA/Mamba causal parity and PI conditioning tests (no fake SSM)."""
from __future__ import annotations

import json
import torch
from torch import nn

from openpi.zeva.behavior_effect import ZevaCTE, ZevaCTEConfig, ZevaPBD, cte_loss


def main():
    torch.manual_seed(17)
    # Different convolution batch shapes may choose different TF32 kernels.
    # Use full FP32 for the sequence/online equivalence audit.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    # Smaller depth/images for a bounded check; real three-stream Mamba kernels.
    cte = ZevaCTE(ZevaCTEConfig(vision_pretrained=False, n_layers=1, image_size=32)).to(device).eval()
    images = torch.rand(2, 4, 3, 3, 32, 32, device=device) * 2 - 1
    actions = torch.randn(2, 3, 15, 16, device=device)
    with torch.no_grad():
        full = cte(images, actions)
        cache = None
        for index in range(3):
            phase, effect, cache = cte.step(images[:, index], None if index == 0 else actions[:, index-1], cache)
            torch.testing.assert_close(phase, full["z_seq"][:, index], rtol=2e-4, atol=2e-4)
            torch.testing.assert_close(effect, full["pred_effect"][:, index], rtol=2e-4, atol=2e-4)
        changed = images.clone()
        changed[:, 2:] = torch.randn_like(changed[:, 2:])
        changed_actions = actions.clone()
        changed_actions[:, 1:] = torch.randn_like(changed_actions[:, 1:]) * 20
        alternate = cte(changed, changed_actions)
        torch.testing.assert_close(full["z_seq"][:, :2], alternate["z_seq"][:, :2])
        torch.testing.assert_close(full["pred_effect"][:, :2], alternate["pred_effect"][:, :2])
        reset_phase, _, _ = cte.step(images[:, 0])
        torch.testing.assert_close(reset_phase, full["z_seq"][:, 0], rtol=2e-4, atol=2e-4)
    cte.train()
    assert all(not m.training for m in cte.vision_encoder.modules() if isinstance(m, nn.BatchNorm2d))
    assert not cte.target_vision_encoder.training
    outputs = cte(images, actions)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], device=device, dtype=torch.bool)
    loss, metrics = cte_loss(outputs, actions, mask, torch.tensor([0, 0], device=device))
    loss.backward()
    assert torch.isfinite(loss)
    assert cte.effect_predictor[-1].weight.grad.norm() > 0
    assert cte.action_predictor[-1].weight.grad.norm() > 0
    assert cte.action_proj.weight.grad.norm() > 0
    assert all(p.grad is None for p in cte.target_vision_encoder.parameters())
    # Padding must not contribute to any of the five losses.
    padded = {k: v.detach().clone() for k, v in outputs.items()}
    for key in ("pred_act", "pred_vis", "target_vis", "pred_effect", "target_effect", "z_global", "z_local"):
        padded[key][1, 2] = 100
    loss_pad, _ = cte_loss(padded, actions, mask, torch.tensor([0, 0], device=device))
    torch.testing.assert_close(loss.detach(), loss_pad)
    pbd = ZevaPBD(prefix_dim=24, expert_dim=12).to(device).eval()
    global_token, phase, effect = [torch.randn(2, 256, device=device) for _ in range(3)]
    prior = pbd.activate(global_token, phase, effect)
    emb = torch.randn(2, 7, 24, device=device)
    pad, att = torch.ones(2, 7, dtype=torch.bool, device=device), torch.zeros(2, 7, dtype=torch.bool, device=device)
    new, new_pad, new_att = pbd.prefix(emb, pad, att)
    assert new.shape == (2, 9, 24) and new_pad[:, :2].all() and not new_att[:, :2].any()
    torch.testing.assert_close(new[:, 2:], emb)
    suffix = torch.randn(2, 50, 12, device=device)
    sp = torch.ones(2, 50, dtype=torch.bool, device=device)
    torch.testing.assert_close(pbd.suffix(suffix, sp, sp, None)[0], suffix)
    nll = pbd.prior_loss(prior, torch.randn(2, 50, 16, device=device))
    pbd.clear()
    # Compare the exact 0.5 inference residual after nonzero projection.
    nn.init.normal_(pbd.prior_emb_proj.weight, std=.01)
    prior = pbd.activate(global_token, phase, effect)
    torch.testing.assert_close(pbd.suffix(suffix, sp, sp, None)[0], suffix + .5*pbd.prior_emb_proj(prior.loc))
    pbd.clear()
    pbd.train()
    prior = pbd.activate(global_token, phase, effect)
    prefix = pbd.prefix(emb, pad, att)[0]
    suffix_on = pbd.suffix(suffix, sp, sp, None)[0]
    objective = prefix.square().mean() + suffix_on.square().mean() + pbd.prior_loss(prior, torch.randn(2, 50, 16, device=device))
    objective.backward()
    assert pbd.global_projector.weight.grad.norm() > 0
    assert pbd.effect_projector.weight.grad.norm() > 0
    assert pbd.action_prior.dist_head[-1].weight.grad.norm() > 0
    pbd.clear()
    torch.testing.assert_close(pbd.prefix(emb, pad, att)[0], emb)
    print(json.dumps({"status": "PASS", "real_mamba_sequence_step_parity": True,
                      "future_image_and_current_action_nonleakage": True, "episode_reset": True,
                      "five_objective_gradients_and_masks": True, "prefix_and_suffix_contract": True,
                      "gaussian_nll": float(nll), "losses": {k:float(v) for k,v in metrics.items()}}), flush=True)


if __name__ == "__main__":
    main()
