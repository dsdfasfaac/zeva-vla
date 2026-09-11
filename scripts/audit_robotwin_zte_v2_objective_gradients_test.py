from __future__ import annotations

import unittest

import torch

try:
    from scripts.audit_robotwin_zte_v2_objective_gradients import _configure_diagnostic_mode
    from scripts.audit_robotwin_zte_v2_objective_gradients import _gradient_snapshot
    from scripts.audit_robotwin_zte_v2_objective_gradients import _pairwise_cosines
except ModuleNotFoundError:  # Direct ``python test.py`` / temp runtime path.
    from audit_robotwin_zte_v2_objective_gradients import _configure_diagnostic_mode
    from audit_robotwin_zte_v2_objective_gradients import _gradient_snapshot
    from audit_robotwin_zte_v2_objective_gradients import _pairwise_cosines


class ObjectiveGradientAuditTests(unittest.TestCase):
    def test_gradient_snapshot_distinguishes_none_from_connected_zero(self):
        used = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        unused = torch.nn.Parameter(torch.tensor([3.0]))
        groups = {"used": [used], "unused": [unused]}
        reports, vectors = _gradient_snapshot((used * 0.0).sum(), groups)

        self.assertEqual(reports["used"]["status"], "zero")
        self.assertEqual(reports["used"]["norm"], 0.0)
        self.assertEqual(reports["unused"]["status"], "none")
        self.assertIsNone(reports["unused"]["norm"])
        self.assertIsNotNone(vectors["used"])
        self.assertIsNone(vectors["unused"])

    def test_pairwise_cosine_marks_zero_and_none_undefined(self):
        vector = torch.tensor([1.0, 0.0])
        zero = torch.zeros(2)
        vectors = {
            "left": {"group": vector},
            "right": {"group": vector},
            "zero": {"group": zero},
            "none": {"group": None},
        }
        result = _pairwise_cosines(vectors, ["left", "right", "zero", "none"], ["group"])
        self.assertAlmostEqual(result["left__vs__right"]["group"]["cosine"], 1.0)
        self.assertEqual(result["left__vs__zero"]["group"]["status"], "undefined_zero_norm")
        self.assertEqual(
            result["left__vs__none"]["group"]["status"],
            "undefined_none_gradient",
        )

    def test_activation_checkpoint_mode_freezes_stochastic_modules(self):
        model = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.MultiheadAttention(4, 2, batch_first=True),
            torch.nn.BatchNorm1d(4),
        )
        config = _configure_diagnostic_mode(model, "activation_checkpoint")
        self.assertTrue(model.training)
        self.assertTrue(config["activation_checkpoint_enabled"])
        self.assertTrue(config["stochastic_modules_disabled"])
        self.assertTrue(config["batchnorm_frozen"])
        self.assertEqual(config["dropout_training_modules"], 0)
        self.assertEqual(config["multihead_attention_training_modules"], 0)
        self.assertEqual(config["batchnorm_training_modules"], 0)


if __name__ == "__main__":
    unittest.main()
