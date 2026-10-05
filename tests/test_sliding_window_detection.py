"""Check the project window grid, source sampling, and hard-vote fusion."""

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sliding_window_detection import (sliding_window_detection,
                                      strict_majority_vote, window_starts)


class SlidingWindowTests(unittest.TestCase):
    def test_paper_window_counts_and_edge_alignment(self):
        for side, expected in ((1024, 8), (512, 45), (256, 209)):
            xs, ys = window_starts(2560, side), window_starts(1440, side)
            self.assertEqual(len(xs) * len(ys), expected)
            self.assertEqual(xs[-1] + side, 2560)
            self.assertEqual(ys[-1] + side, 1440)

    def test_hard_majority_and_ties(self):
        votes = np.array([[0, 1, 1, 2, 2, 3]], dtype=np.uint32)
        count = np.array([[1, 1, 2, 3, 4, 4]], dtype=np.uint32)
        np.testing.assert_array_equal(strict_majority_vote(votes, count),
                                      [[0, 1, 0, 1, 0, 1]])

    def test_source_pixels_and_complete_coverage(self):
        source = np.random.default_rng(0).integers(0, 2, (17, 29), dtype=np.uint8)
        image = Image.fromarray(source)
        for side in (8, 12, 40):
            actual, calls = sliding_window_detection(image, np.asarray, side)
            np.testing.assert_array_equal(actual, source)
            self.assertEqual(calls, len(window_starts(29, side)) * len(window_starts(17, side)))

    def test_agrees_with_original_experiment_loop(self):
        image = Image.new('L', (37, 23))
        side = 12
        def predict(window):
            h, w = window.height, window.width
            return (np.indices((h, w)).sum(axis=0) % 3 == 0).astype(np.uint8)
        votes = np.zeros((23, 37), dtype=np.uint16)
        count = np.zeros_like(votes)
        for y in window_starts(23, side):
            for x in window_starts(37, side):
                votes[y:y + side, x:x + side] += predict(image.crop((x, y, x + side, y + side)))
                count[y:y + side, x:x + side] += 1
        expected = ((votes * 2) > count).astype(np.uint8)
        actual, _ = sliding_window_detection(image, predict, side)
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_window_predictions_are_rejected(self):
        image = Image.new('L', (8, 8))
        for predict in (lambda w: np.ones((1, 1)), lambda w: np.full((8, 8), 255)):
            with self.assertRaises(ValueError):
                sliding_window_detection(image, predict, 8)

    def test_invalid_grid_and_uncovered_pixels_are_rejected(self):
        for overlap in (-0.1, 1, float('nan')):
            with self.assertRaises(ValueError):
                window_starts(10, 4, overlap)
        with self.assertRaises(ValueError):
            strict_majority_vote(np.zeros((2, 2)), np.zeros((2, 2)))


if __name__ == '__main__':
    unittest.main()
