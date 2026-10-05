# Custom sliding-window detection and majority voting

[`src/sliding_window_detection.py`](../src/sliding_window_detection.py) contains the project's external window loop and binary vote fusion. It was extracted from the original [`run_window_experiment.py`](../src/run_window_experiment.py), which now calls the same implementation for the paper's comparison.

## Algorithm

1. Keep the source image at its original pixel sampling.
2. Generate overlapping square windows. The paper uses 50% nominal overlap, so the stride is half the window side.
3. Align the final window with each image boundary to cover the full image. This can increase overlap locally.
4. Run one whole-image model pass on each supplied window and return its binary mask to the window's position on the source grid.
5. Count fracture votes `F` and total window coverage `C` at each pixel. Assign fracture only when `2F > C`; an exact tie is background.

```python
background_votes = coverage - fracture_votes
mask = (fracture_votes > background_votes).astype(np.uint8)
```

The model adapter sets `model.test_cfg.mode = 'whole'` and removes the `Resize` transform. MMSegmentation provides the forward pass for each window. Model-side padding to a multiple of 16 is removed when predictions are returned to the supplied window size.

## Difference from MMSegmentation's internal slide

| Operation | Project implementation | MMSegmentation `EncoderDecoder.slide_inference` |
| --- | --- | --- |
| Window loop | Explicit loop in [`sliding_window_detection`](../src/sliding_window_detection.py) | Loop inside the segmentor |
| Values fused across windows | Hard binary class labels | Class logits |
| Fusion | Strict majority voting | Average logits by pixel coverage, then classify |
| Exact binary tie | Background | Determined by the averaged logits |
| Model mode used by the project algorithm | `whole`, once per external window | `slide` |

The stock logit-averaging implementation can be inspected in [`encoder_decoder.py`](../third_party/mmsegmentation/mmseg/models/segmentors/encoder_decoder.py). The custom hard-vote method is used for the full-view inference-strategy comparison. The separate 65-image validation pipeline preserves its recorded MMSegmentation internal-slide settings.

## Window counts for the paper's source images

For a 2560 × 1440 source image:

| Window side | Nominal stride | Window calls |
| --- | --- | --- |
| 1024 | 512 | 8 |
| 512 | 256 | 45 |
| 256 | 128 | 209 |

These counts exclude any shape warm-up in the timed comparison script. If an image axis is shorter than the selected window side, the reusable function uses the available pixels in one window along that axis without enlarging the image.

## Run on a single image

After following the root README environment setup and setting `PYTHONPATH`, run from the repository root:

```powershell
python 'src/sliding_window_detection.py' --image 'data/full_views/a11.png' --checkpoint 'PATH_TO_OURS_CHECKPOINT.pth' --window-size 512 --overlap 0.5 --output 'outputs/a11_window_vote.png'
```

The output has the same width and height as the source; PNG values are 0 for background and 255 for fracture. Use `--device cpu` for CPU inference. Checkpoints are supplied separately.

## Reuse with another predictor

The core algorithm requires only NumPy and Pillow. The callback accepts a PIL window and returns a same-size NumPy mask with labels 0 and 1:

```python
from sliding_window_detection import sliding_window_detection

mask, window_calls = sliding_window_detection(
    image, predict_window, window_size=512, overlap=0.5)
```

For the complete resize-versus-window comparison, run [`src/run_window_experiment.py`](../src/run_window_experiment.py). It applies the custom method to the three full views and computes per-image mIoU on their original grids. `--baseline` processes each original image once and can run on a fresh checkout.
