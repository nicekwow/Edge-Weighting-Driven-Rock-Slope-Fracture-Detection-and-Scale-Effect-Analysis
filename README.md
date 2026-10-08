# Edge-Weighting-Driven Rock Slope Fracture Detection and Scale Effect Analysis

Grad+U-Net segments visible rock-slope fractures using a U-Net backbone, a pyramid pooling head, and Sobel edge-weighted cross-entropy. This repository provides the training dataset, model configurations, MMSegmentation source, and code for training and sliding-window detection.

## Method

### Sobel edge-weighted loss

The model configuration is [`configs/ours.py`](configs/ours.py). [`SobelImageEdgeAwareLoss`](src/research_extensions/sobel_losses.py) computes edge weights from the aligned training image. RGB intensities are recovered from the normalized input and converted to grayscale as `I = 0.299 R + 0.587 G + 0.114 B`, with intensities in [0, 1]. The horizontal and vertical 3 x 3 Sobel responses give the edge magnitude

$$
E_i = \frac{\sqrt{(G_x * I)_i^2 + (G_y * I)_i^2}}{\sqrt{20}}.
$$

The fixed normalization keeps the weights in [0, 1]. For a fully labelled training crop with `N` pixels and pixel cross-entropy `ell_i`, the decoder objective is

$$
L_{decode} = \frac{1}{N}\sum_i (1 + 10E_i)\ell_i.
$$

The decoder combines cross-entropy with weight 1 and an additional Sobel-weighted term with weight 10. The auxiliary FCN head uses cross-entropy with weight 0.4. [`sobel_image_training.py`](src/research_extensions/sobel_image_training.py) passes the aligned image to the loss without adding trainable parameters. Replicate padding is used for the Sobel convolution, and ignored-label neighbourhoods receive zero additional edge weight.

### Sliding-window detection

[`src/sliding_window_detection.py`](src/sliding_window_detection.py) implements the window grid and majority voting. It divides the original image into overlapping windows without resizing. Each window is processed once, and its binary prediction is returned to the source image grid. The nominal overlap is 50%; the final window is aligned with each image boundary for complete coverage.

At each pixel, `F` counts fracture predictions and `C` counts all covering windows. The fused mask assigns fracture when `2F > C` and background otherwise, including ties. The project method fuses binary predictions; MMSegmentation's built-in sliding inference averages class logits.

[`src/run_window_experiment.py`](src/run_window_experiment.py) compares this method with whole-image resizing. Resize uses a single full-image input with long edge 1024, 512, or 256 pixels and preserves the aspect ratio. Sliding-window detection uses square windows of the same side lengths on the original image. Predictions are evaluated on the original image grid.

## Data

The dataset is stored in [`data/slope_fracture`](data/slope_fracture), relative to the repository root. The training configurations use this directory as `data_root`:

```text
data/slope_fracture/
├── img_dir/
│   ├── train/
│   └── val/
└── ann_dir/
    ├── train/
    └── val/
```

`img_dir` stores images and `ann_dir` stores annotation masks. The `train` and `val` subdirectories are used for training and validation, respectively. Run the commands below from the repository root so these relative paths resolve correctly.

## Installation

The environment uses Python 3.8.18, PyTorch 2.1.0, torchvision 0.16.0, CUDA 11.8, MMCV 2.1.0, MMEngine 0.9.1, and MMSegmentation 1.2.2. Run the following commands in PowerShell from the repository root:

```powershell
conda create -n rock-fracture python=3.8.18 -y
conda activate rock-fracture
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
python -m pip install openmim==0.3.9
mim install "mmengine==0.9.1"
mim install "mmcv==2.1.0"
python -m pip install "numpy==1.24.3" "scipy==1.10.1" "scikit-image==0.21.0" "Pillow==10.0.1" "opencv-python==4.10.0.84"
python -m pip install -e .\third_party\mmsegmentation
$env:PYTHONPATH = (Resolve-Path 'src').Path + ';' + (Resolve-Path 'third_party/mmsegmentation').Path
```

The setup follows the [PyTorch installation instructions](https://docs.pytorch.org/get-started/previous-versions/) and [MMSegmentation installation guide](https://mmsegmentation.readthedocs.io/en/main/get_started.html). GPU training requires an NVIDIA driver compatible with CUDA 11.8. Check the environment and dataset with:

```powershell
python -c "import torch, mmcv, mmengine, mmseg; from mmcv.ops import nms; print(torch.__version__, torch.version.cuda, mmcv.__version__, mmengine.__version__, mmseg.__version__); print(torch.cuda.is_available())"
python 'src/check_data.py'
```

## Usage

### Training and validation

```powershell
python 'third_party/mmsegmentation/tools/train.py' 'configs/ours.py' --work-dir 'outputs/ours'
python 'third_party/mmsegmentation/tools/test.py' 'configs/ours.py' 'PATH_TO_OURS_CHECKPOINT.pth' --work-dir 'outputs/ours_test'
```

Use the checkpoint trained with the selected configuration. [`configs/unet.py`](configs/unet.py) provides the U-Net comparison model; [`deeplabv3plus.py`](configs/deeplabv3plus.py) and [`segmenter.py`](configs/segmenter.py) provide the other comparison models. Training pipelines and optimization settings are defined in each configuration.

### Sliding-window detection

```powershell
python 'src/sliding_window_detection.py' --config 'configs/ours.py' --checkpoint 'PATH_TO_OURS_CHECKPOINT.pth' --image 'PATH_TO_IMAGE.png' --window-size 512 --overlap 0.5 --output 'outputs/fractures.png'
```

The output preserves the source image dimensions, with pixel values 0 for background and 255 for fracture. Add `--device cpu` for CPU inference.

### Scale and inference comparisons

```powershell
python 'src/run_scale_evaluation.py' --checkpoint 'PATH_TO_UNET_CHECKPOINT.pth'
python 'src/run_window_experiment.py' --checkpoint 'PATH_TO_OURS_CHECKPOINT.pth'
python 'src/run_window_experiment.py' --baseline --checkpoint 'PATH_TO_OURS_CHECKPOINT.pth'
```

The baseline processes each original image once. The scale-group evaluation uses the pipeline in [`configs/evaluation_65_images.py`](configs/evaluation_65_images.py).

## Source files

| Path | Function |
| --- | --- |
| [`src/research_extensions/sobel_losses.py`](src/research_extensions/sobel_losses.py) | Sobel kernels, gradient magnitude, and edge-weighted cross-entropy. |
| [`src/research_extensions/sobel_image_training.py`](src/research_extensions/sobel_image_training.py) | Pass aligned image intensities to the decoder loss. |
| [`src/sliding_window_detection.py`](src/sliding_window_detection.py) | Window generation, single-pass prediction, and binary majority voting. |
| [`configs/ours.py`](configs/ours.py) | Grad+U-Net training configuration. |
| [`third_party/mmsegmentation`](third_party/mmsegmentation) | MMSegmentation model source and training/testing entry points. |

Implementation checks cover Sobel responses, normalization, loss gradients, ignored labels, window coverage, and voting ties:

```powershell
python -m unittest discover -s tests -v
```

MMSegmentation is distributed under its [Apache 2.0 license](third_party/mmsegmentation/LICENSE).
