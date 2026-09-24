# Edge-Weighting-Driven Rock Slope Fracture Detection and Scale Effect Analysis

This repository provides image data, annotations, model configurations, and Python code for rock-slope fracture segmentation and inference-scale analysis. Trained checkpoints, prediction outputs, metric tables, figures, logs, and manuscript files are not included.

## Data

| Path | Contents |
| --- | --- |
| `data/slope_fracture` | 264 training and 65 validation image–mask pairs. Mask values are 0 for background and 1 for fracture. |
| `data/small` | 40 validation image–mask pairs in the small-scale group. |
| `data/large` | 25 validation image–mask pairs in the large-scale group. |
| `data/full_views` | Three complete 2560 × 1440 views (`a11`, `a13`, `a14`), LabelMe annotations, and binary reference masks. Mask values are 0 for background and 255 for fracture. |

The `small` and `large` groups partition the same 65 validation images; they are not additional samples. Run `python src/check_data.py` to verify image–mask pairing, dimensions, class values, and byte identity with the validation set. Read the [annotation notes](data/full_views/ANNOTATIONS.md) before using the full-view masks. File counts and sizes are recorded in [`data/dataset_manifest.json`](data/dataset_manifest.json).

## Code and configurations

| Path | Purpose |
| --- | --- |
| `src/research_extensions` | Register the binary dataset and `EdgeAwareLoss` with MMSegmentation. |
| `third_party/mmsegmentation/mmseg` | Core MMSegmentation source package from the local research environment. |
| `third_party/mmsegmentation/tools/train.py` and `test.py` | MMSegmentation training and testing entry points. |
| `configs/unet.py` and `configs/ours.py` | U-Net training configurations without and with edge-aware loss. |
| `configs/deeplabv3plus.py` and `configs/segmenter.py` | Comparison-model training configurations. |
| `configs/evaluation_65_images.py` | Fixed validation pipeline for the two scale groups. |
| `src/run_scale_evaluation.py` | Evaluate the 40/25 split using per-class intersections and unions. |
| `src/run_window_experiment.py` | Compare single-pass whole-image resizing with sliding-window detection on the three full views. |

The project-specific edge-aware loss is implemented in [`src/research_extensions/my_loss.py`](src/research_extensions/my_loss.py). It identifies label boundaries from 3 × 3 neighborhoods and weights cross-entropy at those pixels. The edge-aware U-Net configuration combines ordinary cross-entropy (`loss_weight=1`) with this boundary term (`loss_weight=10`) in `loss_decode`. The extension is registered through `src/research_extensions/__init__.py`.

The model configurations retain the recorded architectures and data pipelines. Their `data_root` values point to this repository, `load_from` is cleared, and `custom_imports` registers the local extension. These path changes do not alter the network or augmentation settings. The included MMSegmentation snapshot retains its [Apache 2.0 license](third_party/mmsegmentation/LICENSE). Its source and local changes are described in the [snapshot notes](third_party/mmsegmentation/README.md). The source environment used Python 3.8.18, PyTorch 2.1.0, torchvision 0.16.0, MMCV 2.1.0, MMEngine 0.9.1, MMSegmentation 1.2.2, NumPy 1.24.3, scikit-image 0.21.0, Pillow 10.0.1, and CUDA 11.8.

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

Supply the checkpoint corresponding to each configuration. Checkpoints are not included in this repository.

```powershell
python 'src/run_scale_evaluation.py' --checkpoint 'PATH_TO_UNET_CHECKPOINT.pth'
python 'src/run_window_experiment.py' --checkpoint 'PATH_TO_OURS_CHECKPOINT.pth'
python 'src/run_window_experiment.py' --baseline --checkpoint 'PATH_TO_OURS_CHECKPOINT.pth'
```

The window experiment compares whole-image long-edge sizes of 1024, 512, and 256 pixels with unscaled square windows of the same side lengths. Windows use 50% nominal overlap. A strict majority vote resolves overlapping predictions, with ties assigned to background. The `--baseline` run processes each original image once. The 65-image evaluation uses a separate resize-and-slide pipeline. Generated predictions and metrics remain in ignored local output directories.
