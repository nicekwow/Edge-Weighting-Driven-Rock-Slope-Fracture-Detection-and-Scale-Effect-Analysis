# Sobel edge-weighted cross-entropy

[`configs/ours.py`](../configs/ours.py) enables the Grad+U-Net model. The edge weights come from the aligned training image, as described in the manuscript: RGB image -> grayscale intensity -> Sobel gradients -> pixel cross-entropy weighting. The segmentation labels provide the classification targets; they do not supply the main model's edge gradients.

## Where to find the method

| Entry | Role |
| --- | --- |
| [`sobel_magnitude`](../src/research_extensions/sobel_losses.py) | Apply the two 3 x 3 Sobel kernels and compute normalized gradient magnitude. |
| `SobelImageEdgeAwareLoss.edge_map` in the same file | Recover RGB intensity from the normalized training input, convert to grayscale, and obtain Sobel weights. |
| `_SobelEdgeAwareLoss.forward` in the same file | Weight pixel cross-entropy and reduce the additional edge term. |
| [`sobel_image_training.py`](../src/research_extensions/sobel_image_training.py) | Pass the aligned training image from the segmentor to the decoder loss. |
| [`configs/ours.py`](../configs/ours.py) | Configure the decoder as CE + 10 x Sobel-weighted CE. |

## Sobel weights

Both gradient components use the standard 3 x 3 kernels:

```text
Gx = [-1  0  1]       Gy = [-1 -2 -1]
     [-2  0  2]            [ 0  0  0]
     [-1  0  1]            [ 1  2  1]
```

The loss reverses the data preprocessor's mean/std normalization and scales RGB intensities to [0, 1]. Grayscale intensity is `I = 0.299 R + 0.587 G + 0.114 B`. These are the geometrically aligned, augmented training images, so the edge weights and classification targets share the same pixel grid.

The normalized Sobel magnitude is

$$E_i = \frac{\sqrt{(G_x * I)_i^2 + (G_y * I)_i^2}}{\sqrt{20}}.$$

The fixed divisor is the maximum magnitude of these kernels over values in [0, 1]. It keeps weights in [0, 1] and preserves differences in contrast between images; no per-image maximum normalization is applied. Replicate padding avoids introducing an artificial edge at the outer image border. Pixels whose 3 x 3 neighbourhood contains ignored labels receive zero additional edge weight.

## Training objective

For a fully labelled image with `N` pixels and pixel cross-entropy `ell_i`, the decoder loss is

$$L_{decode} = \frac{1}{N}\sum_i \ell_i + 10\frac{1}{N}\sum_i E_i\ell_i
            = \frac{1}{N}\sum_i (1 + 10E_i)\ell_i.$$

The configuration uses

```python
loss_decode = [
    dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
    dict(type='SobelImageEdgeAwareLoss', loss_name='loss_edge',
         loss_weight=10.0, edge_weight=1.0, include_base_loss=False,
         avg_non_ignore=True, mean=(123.675, 116.28, 103.53),
         std=(58.395, 57.12, 57.375)),
]
```

Ordinary cross-entropy is counted once. The additional edge term is averaged over valid pixels rather than edge pixels alone. When ignored labels occur, ordinary CE retains the MMSegmentation reduction in the configuration, while the additional term uses the number of valid pixels. The auxiliary FCN head contributes its separate cross-entropy term with weight 0.4. Image weights are detached from the model gradient graph.

## Image-to-loss data flow

[`SobelImageEncoderDecoder` and `SobelImagePSPHead`](../src/research_extensions/sobel_image_training.py) pass the preprocessed training image to the decoder loss. Their backbone and head forward operations are inherited from MMSegmentation. This data flow adds no trainable parameters and preserves the names and shapes of the learned tensors. The model has 29,046,276 trainable parameters.

The image loss assumes RGB input with the mean and standard deviation shown above. If the data preprocessor's normalization or channel order changes, update the loss configuration to match it.

## Configurations

| Configuration | Purpose |
| --- | --- |
| [`ours.py`](../configs/ours.py) | Main Grad+U-Net configuration with image-Sobel edge weights. |
| [`unet.py`](../configs/unet.py) | U-Net comparison model with ordinary cross-entropy. |
| [`ours_sobel_image.py`](../configs/ours_sobel_image.py) | Alias of `ours.py` for existing image-Sobel commands. |
| [`ours_sobel_mask.py`](../configs/ours_sobel_mask.py) | Optional comparison using Sobel gradients of the binary reference mask. |

The optional mask configuration uses the same Sobel kernels, magnitude scaling, seed, architecture, augmentation, optimizer, schedule and validation pipeline as `ours.py`. It overrides the loss source and uses the standard `EncoderDecoder` and `PSPHead`, because image inputs are unnecessary for mask-derived weights. It is not the image-based formulation described in the manuscript.

The main model retains 512 x 512 training crops, random resize augmentation, SGD with learning rate 0.001, 1,000 warm-up iterations, a 160,000-iteration schedule, and seed 1498288285. The supplied `unet.py` retains its separate optimization settings: learning rate 0.01 and no linear warm-up. Both use a U-Net backbone and PSP head with pooling scales `(1, 2, 3, 6)`.
