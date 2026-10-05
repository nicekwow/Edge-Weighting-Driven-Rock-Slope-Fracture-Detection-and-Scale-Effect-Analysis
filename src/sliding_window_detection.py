"""Project implementation of sliding-window detection with hard majority voting.

The window loop operates on the original image. Each window produces one
binary prediction, and overlapping labels are fused by strict majority:
fracture_votes > background_votes. Ties are assigned to background.

The pure NumPy/Pillow functions can be used with any binary predictor.
The CLI uses MMSegmentation only to predict individual windows; it sets
``model.test_cfg.mode = 'whole'`` and removes the image-resize transform.
"""

import argparse
import copy
import math
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def window_starts(length, window_size, overlap=0.5):
    """Return axis positions with nominal overlap and an edge-aligned last window.

    A short image axis uses a single window without enlarging the image.
    Aligning the final window with the boundary can increase its overlap.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError('length must be a positive integer')
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError('window_size must be a positive integer')
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError('overlap must be finite and in [0, 1)')
    if length <= window_size:
        return [0]
    stride = max(1, int(window_size * (1 - overlap)))
    starts = list(range(0, length - window_size + 1, stride))
    if starts[-1] != length - window_size:
        starts.append(length - window_size)
    return starts


def strict_majority_vote(fracture_votes, coverage):
    """Fuse hard labels; an exact tie is background (0).

    ``coverage`` counts all windows covering each pixel. Each binary window
    label contributes either one fracture vote or one background vote.
    """
    fracture_votes, coverage = np.asarray(fracture_votes), np.asarray(coverage)
    if fracture_votes.shape != coverage.shape:
        raise ValueError('fracture_votes and coverage must have the same shape')
    if np.any(coverage <= 0):
        raise ValueError('every pixel must be covered by at least one window')
    if np.any(fracture_votes < 0) or np.any(fracture_votes > coverage):
        raise ValueError('fracture votes must lie between zero and coverage')
    background_votes = coverage - fracture_votes
    return (fracture_votes > background_votes).astype(np.uint8)


def sliding_window_detection(image, predict_window, window_size=512, overlap=0.5):
    """Predict native image windows and fuse them with strict majority voting.

    Args:
        image (PIL.Image.Image): Source image, without global resizing.
        predict_window (callable): Accepts a PIL window and returns a binary
            NumPy mask of the same height and width (0 background, 1 fracture).
        window_size (int): Square window side in source pixels.
        overlap (float): Nominal fractional overlap. The paper uses 0.5.

    Returns:
        tuple: Binary mask on the source grid and number of predictor calls.
    """
    width, height = image.size
    xs = window_starts(width, window_size, overlap)
    ys = window_starts(height, window_size, overlap)
    fracture_votes = np.zeros((height, width), dtype=np.uint32)
    coverage = np.zeros((height, width), dtype=np.uint32)
    calls = 0
    for y in ys:
        for x in xs:
            right, bottom = min(x + window_size, width), min(y + window_size, height)
            window = image.crop((x, y, right, bottom))
            labels = np.asarray(predict_window(window))
            if labels.shape != (bottom - y, right - x):
                raise ValueError('window prediction must match the window height and width')
            if not np.all((labels == 0) | (labels == 1)):
                raise ValueError('window prediction must contain binary labels 0 and 1')
            fracture_votes[y:bottom, x:right] += labels.astype(np.uint32)
            coverage[y:bottom, x:right] += 1
            calls += 1
    return strict_majority_vote(fracture_votes, coverage), calls


def build_single_pass_model(config, checkpoint, device='cuda:0'):
    """Build a predictor that processes each supplied image exactly once.

    Model-side padding to a multiple of 16 is retained. No resize transform
    or MMSegmentation internal slide is used for these native windows.
    """
    from mmengine import Config
    from mmseg.apis import init_model
    import research_extensions  # noqa: F401  Register project loss and dataset.

    cfg = Config.fromfile(str(config))
    cfg.load_from = None
    cfg.model.test_cfg = dict(mode='whole')
    cfg.model.data_preprocessor.test_cfg = dict(size_divisor=16)
    cfg.test_pipeline = [copy.deepcopy(t) for t in cfg.test_pipeline
                         if t['type'] != 'Resize']
    model = init_model(cfg, str(checkpoint), device=device).eval()
    if model.test_cfg['mode'] != 'whole':
        raise RuntimeError('custom sliding-window detection requires single-pass windows')
    return model


def predict_mask(model, image):
    """Return a hard binary mask from one MMSegmentation forward pass."""
    from mmseg.apis import inference_model

    # Pillow reads RGB; MMSegmentation inference receives a BGR NumPy array.
    bgr = np.asarray(image)[:, :, ::-1].copy()
    prediction = inference_model(model, bgr).pred_sem_seg.data[0]
    return prediction.cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs' / 'ours.py')
    parser.add_argument('--window-size', type=int, default=512)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0', help='For example cuda:0 or cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'fractures.png')
    args = parser.parse_args()
    for name in ('image', 'checkpoint', 'config'):
        if not getattr(args, name).is_file():
            parser.error(f'{name} not found: {getattr(args, name)}')
    if args.window_size <= 0 or not math.isfinite(args.overlap) or not 0 <= args.overlap < 1:
        parser.error('window size must be positive and overlap must be in [0, 1)')
    with Image.open(args.image) as source:
        image = source.convert('RGB')
    model = build_single_pass_model(args.config, args.checkpoint, args.device)
    mask, calls = sliding_window_detection(
        image, lambda window: predict_mask(model, window), args.window_size, args.overlap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask * 255).save(args.output)
    print(f'Saved {args.output}: {image.width} x {image.height}, {calls} windows, '
          'strict majority voting (ties = background)')


if __name__ == '__main__':
    main()
