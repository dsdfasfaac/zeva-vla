from __future__ import annotations

import unittest

import torch

from openpi.zeva.cte_eap import ZevaEffectActionPrior
from openpi.zeva.pim_policy import AttemptPersistentMemory, ZevaPIMEAP
from scripts.build_robotwin_pim_artifacts import build_pairings


class PIMMemoryTest(unittest.TestCase):
    def test_attempt_and_episode_reset_contract(self):
        memory = AttemptPersistentMemory(max_attempts=2, max_entries_per_attempt=2)
        self.assertEqual(memory.snapshot(), {"bit_entries": 0, "pim_attempts": 0, "pim_entries": 0})
        for value in range(3):
            memory.append_bit(torch.full((1, 256), value), torch.full((1, 256), value + 10))
        self.assertIsNone(memory.entries()[0])
        memory.reset_attempt()
        phase, bit = memory.entries()
        self.assertEqual(tuple(phase.shape), (1, 2, 256))
        self.assertEqual(tuple(bit.shape), (1, 2, 256))
        self.assertEqual(memory.snapshot()["pim_attempts"], 1)
        memory.append_bit(torch.zeros(1, 256), torch.ones(1, 256))
        memory.reset_attempt()
        self.assertEqual(memory.snapshot()["pim_attempts"], 2)
        memory.reset_episode()
        self.assertEqual(memory.snapshot(), {"bit_entries": 0, "pim_attempts": 0, "pim_entries": 0})


class PIMEAPTest(unittest.TestCase):
    def test_first_attempt_is_exact_parent_path(self):
        torch.manual_seed(7)
        parent = ZevaEffectActionPrior(dim=256, prefix_dim=32, expert_dim=16).eval()
        pim = ZevaPIMEAP(dim=256, prefix_dim=32, expert_dim=16).eval()
        pim.load_state_dict(parent.state_dict(), strict=False)
        global_token, phase, bit = (torch.randn(2, 256) for _ in range(3))
        parent_prior = parent.activate(global_token, phase, bit)
        parent_tokens, parent_residual = (value.clone() for value in parent._active)
        parent.clear()
        pim_prior = pim.activate(global_token, phase, bit, include_pim=False)
        pim_tokens, pim_residual = pim._active
        self.assertTrue(torch.equal(parent_tokens, pim_tokens))
        self.assertTrue(torch.equal(parent_residual, pim_residual))
        self.assertTrue(torch.equal(parent_prior.loc, pim_prior.loc))
        self.assertTrue(torch.equal(parent_prior.scale, pim_prior.scale))

    def test_cross_attempt_memory_adds_one_prefix(self):
        module = ZevaPIMEAP(dim=256, prefix_dim=32, expert_dim=16).eval()
        current = torch.randn(2, 256)
        module.activate(
            torch.randn(2, 256), current, torch.randn(2, 256),
            pim_phase=torch.randn(2, 5, 256), pim_bit=torch.randn(2, 5, 256),
            pim_mask=torch.ones(2, 5, dtype=torch.bool),
        )
        tokens, residual = module._active
        self.assertEqual(tuple(tokens.shape), (2, 3, 32))
        self.assertEqual(tuple(residual.shape), (2, 50, 16))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_pim_route_has_nonzero_gradients(self):
        module = ZevaPIMEAP(dim=256, prefix_dim=32, expert_dim=16).train()
        prior = module.activate(
            torch.randn(2, 256), torch.randn(2, 256), torch.randn(2, 256),
            pim_phase=torch.randn(2, 5, 256), pim_bit=torch.randn(2, 5, 256),
            pim_mask=torch.ones(2, 5, dtype=torch.bool),
        )
        tokens, residual = module._active
        target = torch.randn_like(prior.loc)
        loss = -prior.log_prob(target).mean() + tokens.square().mean() + residual.square().mean()
        loss.backward()
        for name, parameter in module.named_parameters():
            if name.startswith(("pim_phase.", "pim_bit.", "pim_query.", "pim_projector.", "pim_to_global.")):
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(float(parameter.grad.float().norm()), 0.0, name)


class PairingTest(unittest.TestCase):
    @staticmethod
    def row(value):
        phase = torch.zeros(2, 256)
        phase[0, 0] = value
        phase[0, 1] = 1
        return {"phase": phase, "effect": phase.clone(), "frames": torch.tensor([0, 15])}

    def test_pairings_are_train_only_and_not_self(self):
        artifact = {"splits": {
            "train": {
                "Clean:task:0": self.row(1), "Clean:task:1": self.row(2),
                "Randomized:task:2": self.row(3), "Randomized:task:3": self.row(4),
            },
            "validation": {"Clean:task:4": self.row(5)},
        }}
        pairings = build_pairings(artifact)
        train_ids = set(artifact["splits"]["train"])
        for split in pairings.values():
            for target, modes in split.items():
                for mode in ("matched", "same_condition_far", "cross_condition"):
                    self.assertIn(modes[mode], train_ids)
                    self.assertNotEqual(modes[mode], target)


if __name__ == "__main__":
    unittest.main()
