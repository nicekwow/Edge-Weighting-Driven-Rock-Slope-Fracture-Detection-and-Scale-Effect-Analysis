# Edge-Weighting-Driven Rock Slope Fracture Detection and Scale Effect Analysis

This repository contains the image data, annotations, configuration files, and Python code for rock-slope fracture segmentation and inference-scale analysis. It does not contain trained checkpoints, prediction outputs, metric tables, figures, logs, or manuscript files.

## Data

| Path | Contents |
| --- | --- |
| `02_实验数据/slope_fracture` | 264 training and 65 validation JPEG/PNG image–mask pairs. In these masks, 0 is background and 1 is fracture. |
| `02_实验数据/small` | Forty validation image–mask pairs assigned to the small-scale group. |
| `02_实验数据/large` | Twenty-five validation image–mask pairs assigned to the large-scale group. |
| `02_实验数据/三幅重建图像与标注` | Three complete 2560 × 1440 views (`a11`, `a13`, `a14`), LabelMe JSON annotations, and binary reference masks. In these masks, 0 is background and 255 is fracture. |

The `small` and `large` groups are a non-overlapping partition of the same 65 validation images, not additional samples. Run `python 03_程序/check_data.py` to check image–mask pairing, dimensions, class values, and byte identity with the validation split. See [annotation provenance](02_实验数据/三幅重建图像与标注/说明.md) before using the three full-view masks.

## Code and configurations

| Path | Purpose |
| --- | --- |
| `03_程序/research_extensions` | Registers the binary dataset and `EdgeAwareLoss` with MMSegmentation. |
| `04_模型配置与日志/work_dirs/*/*.py` | Four model configuration snapshots. The edge-aware U-Net has an additional run-specific config one directory deeper. |
| `03_程序/configs/evaluation_65_images.py` | Fixed validation pipeline for the two scale groups. |
| `03_程序/run_scale_evaluation.py` | Evaluate the 40/25 split, accumulating per-class intersections and unions. |
| `03_程序/run_window_experiment.py` | Compare single-pass whole-image resizing with sliding-window detection on the three complete views. |
| `03_程序/orientation.py` | Skeleton-path extraction and length-weighted image-plane direction distributions. |
| `03_程序/make_figures13_16.py` | Build tables and figures from predictions created locally. |
| `03_程序/evaluate_saved_masks.py` | Recompute pixel metrics from locally saved masks. |
| `03_程序/virtual_camera_fracture.py` | Virtual-camera and trace-analysis source code; Blender scene assets are supplied separately. |

The published configuration copies preserve the model architectures and data pipelines. Their `data_root` points to this repository, `load_from` is cleared, and `custom_imports` registers the local extension. These are portability changes; they do not change the recorded network or augmentation settings. The source environment used Python 3.8.18, PyTorch 2.1.0, torchvision 0.16.0, MMCV 2.1.0, MMEngine 0.9.1, MMSegmentation 1.2.2, NumPy 1.24.3, scikit-image 0.21.0, Pillow 10.0.1, and CUDA 11.8.

## Running the analyses

From the repository root, make `03_程序` importable so the custom classes can be registered. In PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path '03_程序').Path
python '03_程序/check_data.py'
```

Supply the checkpoint corresponding to each configuration. The validation script uses the cross-entropy U-Net checkpoint; the three-view strategy comparison uses the edge-aware U-Net checkpoint. The scripts accept checkpoint paths without requiring the original directory layout:

```powershell
python '03_程序/run_scale_evaluation.py' --checkpoint 'PATH_TO_CE_UNET_CHECKPOINT.pth'
python '03_程序/run_window_experiment.py' --checkpoint 'PATH_TO_EDGE_AWARE_UNET_CHECKPOINT.pth'
python '03_程序/run_window_experiment.py' --baseline --checkpoint 'PATH_TO_EDGE_AWARE_UNET_CHECKPOINT.pth'
python '03_程序/make_figures13_16.py'
python '03_程序/evaluate_saved_masks.py'
```

The window script compares long-edge whole-image sizes of 1024, 512, and 256 pixels with unscaled square windows of the same side lengths. Windows have 50% nominal overlap; a strict majority vote resolves overlaps, with ties assigned to background. The `--baseline` run processes each original image once. The 65-image evaluation uses its separately archived resize-and-slide pipeline. Generated predictions, figures, tables, and logs remain local under ignored output directories.

## 中文说明

本仓库仅发布研究数据、标注、配置和程序，不含权重、预测结果、指标表、图件、日志及论文。训练集 264 张、验证集 65 张；`small` 40 张和 `large` 25 张是同一验证集的两个不重叠分组。三幅完整图像的掩码采用 0/255 编码，训练与验证掩码采用 0/1 编码。运行前请阅读[三图标注来源](02_实验数据/三幅重建图像与标注/说明.md)。整图缩放、滑动窗口检测和 65 图分组验证使用的推理流程不同，程序分别给出。
