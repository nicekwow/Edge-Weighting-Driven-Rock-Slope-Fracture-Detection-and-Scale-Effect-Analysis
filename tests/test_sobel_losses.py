"""Check Sobel weight sources, scaling, ignored pixels, and gradients."""

import itertools
import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party' / 'mmsegmentation'))
sys.path.insert(0, str(ROOT / 'src'))
from research_extensions.sobel_losses import (SobelImageEdgeAwareLoss,
    SobelMaskEdgeAwareLoss, get_sobel_mask_edge_map, sobel_magnitude)


class SobelLossTests(unittest.TestCase):
    def test_fixed_normalization_for_all_binary_neighbourhoods(self):
        patterns = torch.tensor(list(itertools.product((0., 1.), repeat=9))).reshape(-1, 1, 3, 3)
        centres = sobel_magnitude(patterns)[:, 1, 1]
        self.assertAlmostEqual(centres.max().item(), 1.0, places=6)
        self.assertGreaterEqual(centres.min().item(), 0)

    def test_known_vertical_boundary(self):
        target = torch.zeros((1, 5, 5), dtype=torch.long)
        target[:, :, 2:] = 1
        expected = torch.zeros((1, 5, 5))
        expected[:, :, 1:3] = 4 / math.sqrt(20)
        torch.testing.assert_close(get_sobel_mask_edge_map(target), expected)

    def test_image_weights_use_image_contrast_and_reverse_normalization(self):
        loss = SobelImageEdgeAwareLoss()
        target = torch.zeros((1, 5, 5), dtype=torch.long)
        raw = torch.zeros((1, 3, 5, 5))
        raw[:, :, :, 2:] = 255
        normalized = (raw - loss.image_mean) / loss.image_std
        image_edges = loss.edge_map(target, 255, normalized)
        expected = torch.zeros((1, 5, 5))
        expected[:, :, 1:3] = 4 / math.sqrt(20)
        torch.testing.assert_close(image_edges, expected)
        self.assertEqual(get_sobel_mask_edge_map(target).sum().item(), 0)
        low_contrast = (raw * 0.25 - loss.image_mean) / loss.image_std
        torch.testing.assert_close(loss.edge_map(target, 255, low_contrast), expected * 0.25)

    def test_flat_image_does_not_follow_reference_mask_edges(self):
        loss = SobelImageEdgeAwareLoss()
        target = torch.zeros((1, 5, 5), dtype=torch.long)
        target[:, :, 2:] = 1
        raw = torch.full((1, 3, 5, 5), 128.)
        image = (raw - loss.image_mean) / loss.image_std
        torch.testing.assert_close(loss.edge_map(target, 255, image),
                                   torch.zeros((1, 5, 5)), atol=1e-7, rtol=0)
        self.assertGreater(get_sobel_mask_edge_map(target).sum().item(), 0)

    def test_losses_backpropagate_finite_gradients(self):
        target = torch.zeros((1, 5, 5), dtype=torch.long)
        target[:, :, 2:] = 1
        image_loss = SobelImageEdgeAwareLoss(loss_weight=10)
        raw = target[:, None].expand(-1, 3, -1, -1).float() * 255
        normalized = (raw - image_loss.image_mean) / image_loss.image_std
        for loss, kwargs in ((SobelMaskEdgeAwareLoss(loss_weight=10), {}),
                             (image_loss, dict(image_inputs=normalized))):
            pred = torch.randn((1, 2, 5, 5), requires_grad=True)
            value = loss(pred, target, **kwargs)
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(pred.grad).all())
            self.assertGreater(pred.grad.abs().sum().item(), 0)

    def test_missing_image_and_ignored_targets(self):
        pred = torch.zeros((1, 2, 5, 5))
        target = torch.zeros((1, 5, 5), dtype=torch.long)
        with self.assertRaises(ValueError):
            SobelImageEdgeAwareLoss()(pred, target)
        target.fill_(255)
        self.assertEqual(SobelMaskEdgeAwareLoss()(pred, target, ignore_index=255).item(), 0)
        image_loss = SobelImageEdgeAwareLoss()
        self.assertEqual(image_loss(pred, target, ignore_index=255,
                                    image_inputs=torch.zeros((1, 3, 5, 5))).item(), 0)


if __name__ == '__main__':
    unittest.main()
