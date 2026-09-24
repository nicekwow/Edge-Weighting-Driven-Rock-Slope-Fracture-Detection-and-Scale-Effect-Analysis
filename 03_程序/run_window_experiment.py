"""Compare whole-image resizing with sliding-window detection.

Each prediction is returned to the 2560 x 1440 source grid before scoring.
For a size of 512, ``Resize`` means one 512-pixel-long-edge image, whereas
``Sliding-window detection`` means 512 x 512 crops of the unscaled image.
"""

import argparse
import csv
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]

# Importing the local extension registers the loss and dataset classes named
# in the archived MMSegmentation configuration.
import research_extensions  # noqa: F401

from mmengine import Config
from mmseg.apis import inference_model, init_model


HERE = ROOT / "05_评价结果" / "图13至16"
SOURCES = ROOT / "02_实验数据" / "三幅重建图像与标注"
PREDICTIONS = HERE / "predictions"
PREDICTIONS.mkdir(parents=True, exist_ok=True)
BASE = ROOT / "04_模型配置与日志" / "work_dirs" / "unet-s5-d16_pspnet_4xb4-160k_mydata_BoundaryLoss-512x512" / "20250519_160132"
CONFIG = BASE / "unet-s5-d16_pspnet_4xb4-160k_mydata_BoundaryLoss-512x512.py"
CHECKPOINT = BASE / "iter_160000.pth"
SID_TO_NAME = {"a11": "Image a", "a13": "Image b", "a14": "Image c"}


def positions(length, side):
    """Cover one axis with 50% nominal overlap and an edge-aligned last crop."""
    stride = side // 2
    starts = list(range(0, length - side + 1, stride))
    if starts[-1] != length - side:
        starts.append(length - side)
    return starts


def predict(model, image):
    """MMSeg expects BGR arrays; the source PNGs are read as RGB by Pillow."""
    bgr = np.asarray(image)[:, :, ::-1].copy()
    return inference_model(model, bgr).pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)


def run_condition(model, image, strategy, side, original_pipeline, native_pipeline):
    """Run one condition, excluding shape warm-up from synchronized timing."""
    width, height = image.size
    if strategy == "Resize":
        pipeline = copy.deepcopy(original_pipeline)
        for transform in pipeline:
            if transform["type"] == "Resize":
                transform["scale"] = (side, side)
                transform["keep_ratio"] = True
        model.cfg.test_pipeline = pipeline
    else:
        model.cfg.test_pipeline = copy.deepcopy(native_pipeline)
    first = image.crop((0, 0, side, side)) if strategy == "Sliding-window detection" else image
    predict(model, first)  # Shape warm-up; excluded from timing.
    torch.cuda.synchronize()
    started = time.perf_counter()
    if strategy == "Native full":
        mask = predict(model, image)
        calls = 1
    elif strategy == "Resize":
        mask = predict(model, image)
        calls = 1
    elif strategy == "Sliding-window detection":
        # Strict majority voting assigns exact ties to background. This
        # differs from averaging class probabilities in MMSeg's internal slide.
        votes = np.zeros((height, width), dtype=np.uint16)
        count = np.zeros((height, width), dtype=np.uint16)
        xs, ys = positions(width, side), positions(height, side)
        for y in ys:
            for x in xs:
                region = predict(model, image.crop((x, y, x + side, y + side)))
                votes[y:y + side, x:x + side] += region
                count[y:y + side, x:x + side] += 1
        assert np.all(count > 0)
        mask = ((votes * 2) > count).astype(np.uint8)
        calls = len(xs) * len(ys)
    else:
        raise ValueError(strategy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert mask.shape == (height, width)
    return mask, calls, elapsed


def evaluate(pred, reference):
    """Return class IoUs and their arithmetic mean on the original grid."""
    p, r = pred.astype(bool), reference.astype(bool)
    tp = int(np.logical_and(p, r).sum())
    fp = int(np.logical_and(p, ~r).sum())
    fn = int(np.logical_and(~p, r).sum())
    tn = int(np.logical_and(~p, ~r).sum())
    fracture_iou = tp / (tp + fp + fn)
    background_iou = tn / (tn + fp + fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                fracture_iou_pct=100 * fracture_iou,
                background_iou_pct=100 * background_iou,
                miou_pct=50 * (fracture_iou + background_iou),
                predicted_area_pct=100 * p.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Run only the three native whole-image inputs")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT,
                        help="Path to the edge-aware U-Net checkpoint")
    parser.add_argument("--config", type=Path, default=CONFIG,
                        help="Path to the corresponding MMSegmentation config")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not args.config.is_file():
        parser.error(f"config not found: {args.config}")
    cfg = Config.fromfile(str(args.config))
    cfg.load_from = None  # The explicit --checkpoint supplies model weights.
    cfg.model.test_cfg = dict(mode="whole")
    cfg.model.data_preprocessor.test_cfg = dict(size_divisor=16)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    model = init_model(cfg, str(args.checkpoint), device="cuda:0").eval()
    original_pipeline = copy.deepcopy(model.cfg.test_pipeline)
    native_pipeline = [copy.deepcopy(t) for t in original_pipeline if t["type"] != "Resize"]
    assert model.test_cfg["mode"] == "whole"
    conditions = ([("Native full", 0)] if args.baseline else
                  [("Resize", s) for s in (1024, 512, 256)] +
                  [("Sliding-window detection", s) for s in (1024, 512, 256)])
    rows = json.loads((HERE / "per_image_metrics.json").read_text(encoding="utf-8")) if args.baseline else []
    for sid, label in SID_TO_NAME.items():
        image = Image.open(SOURCES / f"{sid}.png").convert("RGB")
        reference = np.asarray(Image.open(SOURCES / f"{sid}_mask.png")) != 0
        for strategy, side in conditions:
            print(f"{label}: {strategy} {side}", flush=True)
            mask, calls, elapsed = run_condition(model, image, strategy, side, original_pipeline, native_pipeline)
            stem = f"{sid}__{strategy.lower().replace(' ', '_').replace('-', '_')}__{side or 'full'}"
            Image.fromarray(mask * 255).save(PREDICTIONS / f"{stem}.png")
            row = dict(image=label, sid=sid, strategy=strategy, size_px=side,
                       forward_calls=calls, time_s=elapsed,
                       reference_area_pct=100 * reference.mean(),
                       **evaluate(mask, reference))
            rows.append(row)
            print(f"  mIoU={row['miou_pct']:.4f}% time={elapsed:.3f}s calls={calls}", flush=True)
            with (HERE / "per_image_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            (HERE / "per_image_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
