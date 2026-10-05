# Edge-weighted cross-entropy

The decoder combines ordinary cross-entropy with an additional edge-weighted term. The configurations differ in the source of the edge map:

| Configuration | Edge source | Implementation |
| --- | --- | --- |
| [`ours_sobel_image.py`](../configs/ours_sobel_image.py) | Sobel magnitude of the aligned training RGB image | `SobelImageEdgeAwareLoss` in [`sobel_losses.py`](../src/research_extensions/sobel_losses.py) |
| [`ours_sobel_mask.py`](../configs/ours_sobel_mask.py) | Sobel magnitude of the binary reference mask | `SobelMaskEdgeAwareLoss` in the same file |
| [`ours.py`](../configs/ours.py) | Categorical boundaries in the binary reference mask | `EdgeAwareLoss` in [`my_loss.py`](../src/research_extensions/my_loss.py) |

`ours.py` preserves the recorded implementation. Its boundary detector does not use Sobel. The Sobel configurations define new training comparisons and do not reproduce the recorded run's weights or scores.

## Sobel edge weights

Both Sobel variants use the same 3 × 3 kernels:

```text
Gx = [-1  0  1]       Gy = [-1 -2 -1]
     [-2  0  2]            [ 0  0  0]
     [-1  0  1]            [ 1  2  1]
```

For an intensity field `I` in [0, 1], the edge magnitude is

$$E_i = \frac{\sqrt{(G_x * I)_i^2 + (G_y * I)_i^2}}{\sqrt{20}}.$$

The fixed divisor is the maximum magnitude of these kernels over input values in [0, 1]. It keeps weights in [0, 1] and preserves contrast differences between images. No per-image maximum normalization is applied. Replicate padding avoids introducing an edge at the outside border. Pixels whose 3 × 3 neighbourhood contains ignored labels receive zero additional edge weight.

For mask weighting, `I` is the binary reference mask. For image weighting, `I` is grayscale intensity computed as `0.299 R + 0.587 G + 0.114 B`. The image is the geometrically aligned, augmented training input. The loss reverses the data preprocessor's mean/std normalization before converting RGB intensities to [0, 1]. Changing the preprocessor's mean/std or channel order requires updating the image-loss configuration.

For a fully labelled image with `N` pixels, the Sobel decoder objective is

$$L_{decode} = \frac{1}{N}\sum_i \ell_i + 10\frac{1}{N}\sum_i E_i\ell_i.$$

Ordinary cross-entropy is counted once: the additional loss uses `include_base_loss=False`, `edge_weight=1`, and `loss_weight=10`. The edge term is averaged over all valid pixels rather than edge pixels alone. With ignored labels, ordinary CE retains its recorded MMSegmentation reduction, while the additional term uses the number of valid pixels. The auxiliary FCN head retains its CE term with weight 0.4.

## Image-to-loss data flow

Image weighting requires the decoder loss to receive image intensities as well as predictions and labels. [`sobel_image_training.py`](../src/research_extensions/sobel_image_training.py) provides `SobelImageEncoderDecoder` and `SobelImagePSPHead` for this training data flow. Their backbone, head forward operations, trainable parameter names and inference operations are inherited unchanged. Image weights are detached from the model gradient graph. The mask variant uses the original `EncoderDecoder` and `PSPHead` directly.

Both Sobel configurations inherit [`ours.py`](../configs/ours.py), retain its 512 × 512 training crops, resize augmentation, SGD, learning-rate schedule and validation pipeline, and set the same seed (`1498288285`). The image variant additionally changes the registered segmentor/head types to pass the image to the loss. Both retain 29,046,276 trainable parameters.

## Recorded categorical boundary weights

For a categorical reference mask, a pixel belongs to the boundary map `B` when its valid 3 × 3 neighbourhood contains more than one class. `get_categorical_edge_map` computes this condition with local maximum and minimum label values. Replicate padding avoids creating a boundary at the outside of the image. Neighbourhoods containing ignored labels are excluded.

```python
edge_map = (local_max != local_min) & all_valid
```

The inputs to this boundary detector are segmentation labels. This implementation uses a binary neighbourhood boundary map; it does not compute image-intensity Sobel gradients.

## Recorded training objective

The decoder uses two separately registered losses:

```python
loss_decode = [
    dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
    dict(type='EdgeAwareLoss', loss_name='loss_edge', loss_weight=10.0,
         edge_weight=1.0, include_base_loss=False, avg_non_ignore=True),
]
```

`EdgeAwareLoss` returns the additional boundary term, so ordinary cross-entropy is counted once. For a fully labelled image with `N` pixels, the configured decoder objective is

$$L_{decode} = \frac{1}{N}\sum_i \ell_i + 10\frac{1}{N}\sum_i B_i\ell_i,$$

where `ell_i` is cross-entropy and `B_i` is 1 at a label boundary and 0 elsewhere. Boundary pixels therefore contribute 11 times their ordinary pixel loss, while non-boundary pixels retain their ordinary contribution. The boundary term is averaged over all valid pixels, rather than over boundary pixels alone.

When ignored labels are present, the boundary term uses the number of non-ignored labels as its averaging factor. The ordinary cross-entropy term retains MMSegmentation's configured reduction. The auxiliary FCN head contributes a separate cross-entropy term with weight 0.4.

## Where to inspect the implementation

| Entry | Role |
| --- | --- |
| `get_categorical_edge_map` in [`my_loss.py`](../src/research_extensions/my_loss.py) | Build the boundary map from reference labels. |
| `edge_aware_loss` in the same file | Multiply pixel cross-entropy by the boundary map and reduce the additional term. |
| `EdgeAwareLoss.forward` in the same file | Apply `loss_weight` and expose `loss_edge` to MMSegmentation. |
| [`research_extensions/__init__.py`](../src/research_extensions/__init__.py) | Register the project extension. |
| [`configs/ours.py`](../configs/ours.py) | Enable CE + 10 × boundary CE. |

The copy under `third_party/mmsegmentation/mmseg/models/losses/my_loss.py` is retained with the source snapshot. The project extension above is loaded by `custom_imports` and is the entry used by the published project configurations.

## Recorded model configurations

Both [`unet.py`](../configs/unet.py) and [`ours.py`](../configs/ours.py) use the MMSegmentation U-Net backbone and `PSPHead`, including pyramid pooling scales `(1, 2, 3, 6)`. The recorded settings are:

| Setting | `unet.py` | `ours.py` and both Sobel variants |
| --- | --- | --- |
| Decoder loss | CE | CE + 10 × edge-weighted CE |
| Active optimizer learning rate (`optim_wrapper`) | 0.01 | 0.001 |
| Linear warm-up | None | 1,000 iterations |
| Training duration | 160,000 iterations | 160,000 iterations |

The historical model comparison includes the listed optimization differences as well as the additional loss. The two new Sobel configurations share their optimization settings.
