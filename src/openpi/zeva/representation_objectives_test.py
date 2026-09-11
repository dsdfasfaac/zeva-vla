import unittest

import torch
from torch.nn import functional as F

from openpi.zeva.representation_objectives import causal_effect_contrastive
from openpi.zeva.representation_objectives import global_supervised_contrastive
from openpi.zeva.representation_objectives import variance_covariance_loss


class RepresentationObjectivesTest(unittest.TestCase):
    def test_matching_effects_outperform_shuffled_effects_without_target_gradients(self):
        values = torch.eye(8, requires_grad=True)
        target = torch.eye(8, requires_grad=True)
        correct = causal_effect_contrastive(values, target)
        shuffled = causal_effect_contrastive(values, target.roll(1, 0))
        self.assertLess(correct.item(), shuffled.item())
        correct.backward()
        self.assertIsNotNone(values.grad)
        self.assertIsNone(target.grad)

    def test_global_positive_pairs_are_augmented_episodes(self):
        features = torch.eye(4)
        labels = torch.arange(4)
        correct = global_supervised_contrastive(features, features, labels)
        shuffled = global_supervised_contrastive(features, features.roll(1, 0), labels)
        self.assertLess(correct.item(), shuffled.item())

    def test_isotropic_cloud_beats_collapsed_unit_features(self):
        torch.manual_seed(9)
        diverse = F.normalize(torch.randn(4096, 8), dim=-1)
        collapsed = F.normalize(torch.ones_like(diverse), dim=-1)
        self.assertLess(variance_covariance_loss(diverse).item(), variance_covariance_loss(collapsed).item())


if __name__ == "__main__":
    unittest.main()
