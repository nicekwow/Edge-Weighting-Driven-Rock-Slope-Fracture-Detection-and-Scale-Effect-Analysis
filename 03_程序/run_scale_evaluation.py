"""Evaluate the fixed 65-image validation set by its two scale groups.

The 40 small and 25 large images are the same 65 images in the validation
split. Class intersections and unions are accumulated before computing group
mIoU; the two group scores are not averaged to obtain the combined score.
"""

import argparse
import os
import csv
import json
import time
import hashlib
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '05_评价结果' / '尺度分组65图'
DATA = ROOT / '02_实验数据'
MODEL_DIR = ROOT / '04_模型配置与日志' / 'work_dirs' / 'unet-s5-d16_pspnet_4xb4-40k_mydata-512x512'
CHECKPOINT = MODEL_DIR / 'iter_160000.pth'
CONFIG_FILE = ROOT / '03_程序' / 'configs' / 'evaluation_65_images.py'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(OUT / '_runtime/matplotlib')
os.environ['CUDA_CACHE_PATH'] = str(OUT / '_runtime/cuda')
import research_extensions  # noqa: F401  Register MyDataset and EdgeAwareLoss.

import numpy as np
import torch
from PIL import Image
from mmengine import Config
from mmengine.dataset import pseudo_collate
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import init_model
from mmseg.registry import DATASETS
from mmseg.evaluation.metrics import IoUMetric
from mmseg.utils import register_all_modules


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def status(state, **values):
    record = dict(state=state, timestamp=time.strftime('%Y-%m-%d %H:%M:%S'), **values)
    write_json(OUT / 'progress.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def exact_scores(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    tp = np.diag(a)
    union = a.sum(axis=0) + a.sum(axis=1) - tp
    iou = tp / union
    return dict(mIoU=float(iou.mean() * 100), IoU_background=float(iou[0] * 100), IoU_fracture=float(iou[1] * 100))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT,
                        help='Path to the cross-entropy U-Net checkpoint')
    args = parser.parse_args()
    checkpoint = args.checkpoint
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        parser.error(f'checkpoint not found: {checkpoint}')
    assert CONFIG_FILE.is_file(), CONFIG_FILE
    # The fixed evaluation config was extracted from the original run log.
    # It retains the 2048 x 1024 resize and 512 x 512 slide with 85-pixel stride.
    cfg = Config.fromfile(str(CONFIG_FILE))
    cfg.load_from = None
    assert cfg.model.test_cfg.mode == 'slide'
    assert tuple(cfg.model.test_cfg.crop_size) == (512, 512)
    assert tuple(cfg.model.test_cfg.stride) == (85, 85)
    assert cfg.test_dataloader.dataset.pipeline == cfg.test_pipeline
    register_all_modules()
    assert torch.cuda.is_available(), 'CUDA unavailable'
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = bool(cfg.env_cfg.cudnn_benchmark)

    # These two directories partition the validation split. Hashing each pair
    # prevents a similarly named but different crop from entering evaluation.
    manifests = []
    names_by_group = {}
    for group, expected in [('small', 40), ('large', 25)]:
        root = DATA / group
        images = sorted((root / 'img').glob('*.jpg'))
        masks = sorted((root / 'ann').glob('*.png'))
        assert len(images) == len(masks) == expected, (group, len(images), len(masks))
        assert {p.stem for p in images} == {p.stem for p in masks}
        names_by_group[group] = {p.name for p in images}
        for p in images:
            ann = root / 'ann' / (p.stem + '.png')
            with Image.open(p) as im:
                size = im.size
            with Image.open(ann) as im:
                assert im.size == size
                assert set(np.unique(np.array(im))).issubset({0, 1})
            val_img = DATA / 'slope_fracture/img_dir/val' / p.name
            val_ann = DATA / 'slope_fracture/ann_dir/val' / ann.name
            ih, ah = sha256(p), sha256(ann)
            manifests.append(dict(group=group, image=str(p), annotation=str(ann), width=size[0], height=size[1], image_sha256=ih, annotation_sha256=ah,
                matches_current_val_image=val_img.is_file() and sha256(val_img) == ih,
                matches_current_val_annotation=val_ann.is_file() and sha256(val_ann) == ah))
    assert not (names_by_group['small'] & names_by_group['large'])
    val_names = {p.name for p in (DATA / 'slope_fracture/img_dir/val').glob('*.jpg')}
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), evaluation_config=str(CONFIG_FILE),
        evaluation_config_sha256=sha256(CONFIG_FILE), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), seed=0, model_test_cfg=dict(cfg.model.test_cfg), test_pipeline=[dict(x) for x in cfg.test_pipeline],
        normalization=dict(cfg.model.data_preprocessor), random_test_augmentation=False,
        fusion='Arithmetic mean of overlapping class logits, followed by bilinear restoration to original shape and argmax.',
        metric='MMSeg IoUMetric: accumulate per-class intersections and unions over each group, then average the two class IoUs.',
        combined_filenames_match_current_val=(names_by_group['small'] | names_by_group['large']) == val_names,
        all_files_match_current_val=all(r['matches_current_val_image'] and r['matches_current_val_annotation'] for r in manifests),
        files=manifests)
    write_json(OUT / 'dataset_and_run_manifest.json', metadata)
    status('loading_model', checkpoint_sha256=metadata['checkpoint_sha256'], total_images=len(manifests))
    model = init_model(cfg, str(checkpoint), device='cuda:0')
    model = revert_sync_batchnorm(model).eval()
    assert list(model.dataset_meta['classes']) == ['background', 'crack']
    combined_results = []
    combined_matrix = np.zeros((2, 2), dtype=np.int64)
    summaries = []
    per_image = []
    for group, expected in [('small', 40), ('large', 25)]:
        group_out = OUT / group
        group_out.mkdir(exist_ok=True)
        data_cfg = cfg.test_dataloader.dataset.copy()
        data_cfg.update(data_root=str(DATA / group), data_prefix=dict(img_path='img', seg_map_path='ann'), test_mode=True, img_suffix='.jpg')
        dataset = DATASETS.build(data_cfg)
        assert len(dataset) == expected
        group_cfg = cfg.copy()
        group_cfg.test_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False, sampler=dict(type='DefaultSampler', shuffle=False), dataset=data_cfg)
        group_cfg.load_from = str(checkpoint)
        group_cfg.work_dir = str(group_out)
        group_cfg.dump(str(group_out / 'resolved_evaluation_config.py'))
        metric = IoUMetric(iou_metrics=['mIoU', 'mDice', 'mFscore'], output_dir=str(group_out / 'predictions'))
        metric.dataset_meta = dataset.metainfo
        matrix = np.zeros((2, 2), dtype=np.int64)
        for i in range(len(dataset)):
            item = dataset[i]
            batch = pseudo_collate([item])
            with torch.inference_mode():
                predictions = model.test_step(batch)
            torch.cuda.synchronize()
            sample = predictions[0]
            pred = sample.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.int64)
            gt = sample.gt_sem_seg.data.squeeze().cpu().numpy().astype(np.int64)
            assert pred.shape == gt.shape
            assert set(np.unique(pred)).issubset({0, 1})
            # Rows are reference classes and columns are predicted classes.
            # Pool integer counts across images before computing each group IoU.
            cm = np.bincount((gt * 2 + pred).ravel(), minlength=4).reshape(2, 2)
            matrix += cm
            scores = exact_scores(cm)
            per_image.append(dict(group=group, image=Path(sample.metainfo['img_path']).name,
                original_width=gt.shape[1], original_height=gt.shape[0], input_width=int(item['inputs'].shape[2]), input_height=int(item['inputs'].shape[1]),
                mIoU=scores['mIoU'], IoU_fracture=scores['IoU_fracture'], IoU_background=scores['IoU_background'],
                true_background_pred_background=int(cm[0, 0]), true_background_pred_fracture=int(cm[0, 1]), true_fracture_pred_background=int(cm[1, 0]), true_fracture_pred_fracture=int(cm[1, 1])))
            metric.process(batch, [s.to_dict() for s in predictions])
            if (i + 1) % 5 == 0 or i + 1 == len(dataset):
                status('evaluating', group=group, completed=i + 1, total=len(dataset), elapsed_seconds=round(time.time() - started, 1))
        official = metric.compute_metrics(metric.results)
        exact = exact_scores(matrix)
        rounded_agree = round(exact['mIoU'], 2) == float(official['mIoU'])
        summary = dict(group=group, images=len(dataset), mIoU=float(official['mIoU']), IoU_fracture=round(exact['IoU_fracture'], 2),
            IoU_background=round(exact['IoU_background'], 2), official_metrics={k: float(v) for k, v in official.items()},
            exact_integer_count_scores=exact, confusion_matrix_rows_reference_columns_prediction=matrix.tolist(), exact_count_mIoU_agrees_to_2dp=rounded_agree,
            official_intersection=[sum(float(r[0][c]) for r in metric.results) for c in range(2)],
            official_union=[sum(float(r[1][c]) for r in metric.results) for c in range(2)])
        write_json(group_out / 'metrics.json', summary)
        summaries.append(summary)
        combined_results.extend(metric.results)
        combined_matrix += matrix
        status('group_complete', group=group, images=len(dataset), mIoU=summary['mIoU'], IoU_fracture=summary['IoU_fracture'], IoU_background=summary['IoU_background'])
    metric_all = IoUMetric(iou_metrics=['mIoU', 'mDice', 'mFscore'])
    metric_all.dataset_meta = model.dataset_meta
    all_official = metric_all.compute_metrics(combined_results)
    all_exact = exact_scores(combined_matrix)
    combined = dict(group='combined', images=len(manifests), mIoU=float(all_official['mIoU']), IoU_fracture=round(all_exact['IoU_fracture'], 2), IoU_background=round(all_exact['IoU_background'], 2),
        official_metrics={k: float(v) for k, v in all_official.items()}, exact_integer_count_scores=all_exact,
        confusion_matrix_rows_reference_columns_prediction=combined_matrix.tolist(), exact_count_mIoU_agrees_to_2dp=round(all_exact['mIoU'], 2) == float(all_official['mIoU']))
    summaries.append(combined)
    write_json(OUT / 'all_metrics.json', summaries)
    with (OUT / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'images', 'mIoU', 'IoU_fracture', 'IoU_background'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summaries)
    with (OUT / 'per_image_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    status('complete', results=[{k: r[k] for k in ['group', 'images', 'mIoU', 'IoU_fracture', 'IoU_background']} for r in summaries],
        elapsed_seconds=round(time.time() - started, 1), all_masks_saved=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        status('failed', error=traceback.format_exc())
        raise
