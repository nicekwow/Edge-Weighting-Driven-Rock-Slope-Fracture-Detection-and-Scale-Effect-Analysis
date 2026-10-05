"""Check the custom loss against explicit boundary maps and pixel CE values."""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party' / 'mmsegmentation'))
sys.path.insert(0, str(ROOT / 'src'))
from research_extensions.my_loss import EdgeAwareLoss, get_categorical_edge_map


class EdgeAwareLossTests(unittest.TestCase):
    def test_known_boundary_and_no_artificial_image_border(self):
        target = torch.zeros((1, 5, 5), dtype=torch.long)
        target[:, :, 2:] = 1
        expected = torch.zeros((1, 5, 5))
        expected[:, :, 1:3] = 1
        torch.testing.assert_close(get_categorical_edge_map(target), expected)
        torch.testing.assert_close(get_categorical_edge_map(torch.ones_like(target)),
                                   torch.zeros_like(expected))

    def test_configured_loss_matches_explicit_weighted_ce_and_backpropagates(self):
        target = torch.zeros((1, 5, 5), dtype=torch.long)
        target[:, :, 2:] = 1
        logits = torch.linspace(-1, 1, 50).reshape(1, 2, 5, 5).requires_grad_()
        ce = F.cross_entropy(logits, target, reduction='none')
        extra = EdgeAwareLoss(loss_weight=10)(logits, target)
        expected = ce.mean() + 10 * ce[:, :, 1:3].sum() / 25
        actual = ce.mean() + extra
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_ignored_regions_do_not_create_label_boundaries(self):
        target = torch.ones((1, 5, 5), dtype=torch.long)
        target[:, 2, 2] = 255
        self.assertEqual(get_categorical_edge_map(target, 255).sum().item(), 0)
        ignored = torch.full_like(target, 255)
        value = EdgeAwareLoss()(torch.zeros((1, 2, 5, 5)), ignored, ignore_index=255)
        self.assertEqual(value.item(), 0)


if __name__ == '__main__':
    unittest.main()
