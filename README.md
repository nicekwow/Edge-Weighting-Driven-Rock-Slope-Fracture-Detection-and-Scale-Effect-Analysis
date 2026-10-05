# Edge-Weighting-Driven Rock Slope Fracture Detection and Scale Effect Analysis

This repository provides image data, annotations, model configurations, and Python code for rock-slope fracture segmentation and inference-scale analysis. Trained checkpoints, prediction outputs, metric tables, figures, logs, and manuscript files are not included.

## Project methods

| Method | Implementation | Configuration or command |
| --- | --- | --- |
| Edge-weighted cross-entropy | [Sobel image and mask losses](src/research_extensions/sobel_losses.py); [image-to-loss data flow](src/research_extensions/sobel_image_training.py) | [`ours_sobel_image.py`](configs/ours_sobel_image.py) / [`ours_sobel_mask.py`](configs/ours_sobel_mask.py) |
| Recorded categorical boundary loss | [`EdgeAwareLoss`](src/research_extensions/my_loss.py) | [`ours.py`](configs/ours.py) |
| Custom sliding-window detection | [Native crops and strict majority voting](src/sliding_window_detection.py) | `python src/sliding_window_detection.py --image IMAGE --checkpoint CHECKPOINT` |

The Sobel comparison uses either aligned training-image intensities or binary reference masks to weight pixel cross-entropy. Both configurations retain the recorded edge-weighted model's architecture, augmentation, optimizer and training schedule, with the same random seed. They are new training variants; their results must be evaluated separately from the recorded categorical-boundary run. See the [loss equations and configuration details](docs/edge_aware_loss.md).

The custom sliding-window implementation processes each original-resolution crop once, accumulates its **binary fracture prediction**, and returns fracture only when more than half of the covering windows predict fracture. Ties become background. This is the project's external voting method. MMSegmentation's built-in `slide_inference` averages logits and is a separate implementation. See the [window grid and voting details](docs/sliding_window_detection.md).

## Data

| Path | Contents |
| --- | --- |
| `data/slope_fracture` | 264 training and 65 validation image–mask pairs. Mask values are 0 for background and 1 for fracture. |
| `data/small` | 40 validation image–mask pairs in the small-scale group. |
| `data/large` | 25 validation image–mask pairs in the large-scale group. |
| `data/full_views` | Three complete 2560 × 1440 views (`a11`, `a13`, `a14`), LabelMe annotations, and binary reference masks. Mask values are 0 for background and 255 for fracture. |

The `small` and `large` groups partition the same 65 validation images; they are not additional samples. Run `python src/check_data.py` to verify image–mask pairing, dimensions, class values, and byte identity with the validation set. File counts and sizes are recorded in [`data/dataset_manifest.json`](data/dataset_manifest.json).

## Code and configurations

| Path | Purpose |
| --- | --- |
| `src/research_extensions` | Register the binary dataset, edge losses, and image-loss training adapters with MMSegmentation. |
| `third_party/mmsegmentation/mmseg` | Core MMSegmentation source package from the local research environment. |
| `third_party/mmsegmentation/tools/train.py` and `test.py` | MMSegmentation training and testing entry points. |
| `configs/unet.py` and `configs/ours.py` | U-Net training configurations without and with edge-aware loss. |
| `configs/ours_sobel_image.py` and `configs/ours_sobel_mask.py` | Compare image-Sobel and mask-Sobel weights under the recorded `ours.py` settings. |
| `configs/deeplabv3plus.py` and `configs/segmenter.py` | Comparison-model training configurations. |
| `configs/evaluation_65_images.py` | Fixed validation pipeline for the two scale groups. |
| `src/run_scale_evaluation.py` | Evaluate the 40/25 split using per-class intersections and unions. |
| `src/run_window_experiment.py` | Compare single-pass whole-image resizing with sliding-window detection on the three full views. |
| `src/sliding_window_detection.py` | Standalone custom window detection and binary majority-vote fusion. |

All three edge-loss configurations combine ordinary decoder cross-entropy (`loss_weight=1`) with an additional edge-weighted term (`loss_weight=10`). The recorded [`ours.py`](configs/ours.py) uses categorical label boundaries from 3 × 3 neighborhoods. The two Sobel configurations use normalized gradient magnitudes. The extension is registered through [`src/research_extensions/__init__.py`](src/research_extensions/__init__.py).

The model configurations retain the recorded architectures and data pipelines. Their `data_root` values point to this repository, `load_from` is cleared, and `custom_imports` registers the local extension. These path changes do not alter the network or augmentation settings. The included MMSegmentation snapshot retains its [Apache 2.0 license](third_party/mmsegmentation/LICENSE). Its source and local changes are described in the [snapshot notes](third_party/mmsegmentation/README.md).

## Environment setup

The source environment used Python 3.8.18, PyTorch 2.1.0, torchvision 0.16.0, CUDA 11.8, MMCV 2.1.0, MMEngine 0.9.1, and MMSegmentation 1.2.2. The following Windows PowerShell commands follow the [PyTorch 2.1.0 installation instructions](https://docs.pytorch.org/get-started/previous-versions/) and the [MMSegmentation installation guide](https://mmsegmentation.readthedocs.io/en/main/get_started.html). Run them from the repository root after installing Conda and an NVIDIA driver suitable for CUDA 11.8.

```powershell
conda create -n rock-fracture python=3.8.18 -y
conda activate rock-fracture
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
python -m pip install openmim==0.3.9
mim install "mmengine==0.9.1"
mim install "mmcv==2.1.0"
python -m pip install "numpy==1.24.3" "scipy==1.10.1" "scikit-image==0.21.0" "Pillow==10.0.1" "opencv-python==4.10.0.84"
python -m pip install -e .\third_party\mmsegmentation
```

The editable installation uses the MMSegmentation source included here. If MIM selects an MMCV source archive rather than a compatible wheel, consult the [MMCV installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for build requirements. Check the installed versions and compiled MMCV operations before running the models:

```powershell
python -c "import torch, mmcv, mmengine, mmseg; from mmcv.ops import nms; print(torch.__version__, torch.version.cuda, mmcv.__version__, mmengine.__version__, mmseg.__version__)"
python 'src/check_data.py'
```

The version check should print `2.1.0 11.8 2.1.0 0.9.1 1.2.2`. GPU training also requires `torch.cuda.is_available()` to return `True`.

## Running the analyses

From the repository root, add both the project code and the included MMSegmentation source to `PYTHONPATH`. In PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path + ';' + (Resolve-Path 'third_party/mmsegmentation').Path
python 'src/check_data.py'
```

The included MMSegmentation entry points can train or test a model with these configurations. Their output directories are ignored by Git. For example:

```powershell
python 'third_party/mmsegmentation/tools/train.py' 'configs/ours.py' --work-dir 'outputs/ours'
python 'third_party/mmsegmentation/tools/test.py' 'configs/ours.py' 'PATH_TO_OURS_CHECKPOINT.pth' --work-dir 'outputs/ours_test'
```

Run the two Sobel variants sequentially on a single GPU:

```powershell
python 'third_party/mmsegmentation/tools/train.py' 'configs/ours_sobel_mask.py' --work-dir 'outputs/ours_sobel_mask'
python 'third_party/mmsegmentation/tools/train.py' 'configs/ours_sobel_image.py' --work-dir 'outputs/ours_sobel_image'
```

Supply the checkpoint corresponding to each configuration. Checkpoints are not included in this repository.

For custom sliding-window detection of one full image:

```powershell
python 'src/sliding_window_detection.py' --config 'configs/ours.py' --checkpoint 'PATH_TO_OURS_CHECKPOINT.pth' --image 'data/full_views/a11.png' --window-size 512 --overlap 0.5 --output 'outputs/a11_mask.png'
```

For the scale groups and full-image inference comparison:

```powershell
python 'src/run_scale_evaluation.py' --checkpoint 'PATH_TO_UNET_CHECKPOINT.pth'
python 'src/run_window_experiment.py' --checkpoint 'PATH_TO_OURS_CHECKPOINT.pth'
python 'src/run_window_experiment.py' --baseline --checkpoint 'PATH_TO_OURS_CHECKPOINT.pth'
```

The window experiment compares whole-image long-edge sizes of 1024, 512, and 256 pixels with unscaled square windows of the same side lengths. Windows use 50% nominal overlap. A strict majority vote resolves overlapping predictions, with ties assigned to background. The `--baseline` run processes each original image once. The 65-image evaluation uses a separate resize-and-slide pipeline. Generated predictions and metrics remain in ignored local output directories.

## Implementation checks

With the environment and `PYTHONPATH` above, run `python -m unittest discover -s tests -v`. The checks cover known Sobel responses, image normalization, finite loss gradients, ignored labels, crop coverage, boundary alignment, and strict voting including ties. They also compare the shared voting function with the original experiment loop.
