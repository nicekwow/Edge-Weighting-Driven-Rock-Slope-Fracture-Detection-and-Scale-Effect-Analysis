# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import warnings
from pathlib import Path
from typing import Optional, Union

import mmcv
import numpy as np
import torch
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import mkdir_or_exist

from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList, dataset_aliases, get_classes, get_palette
from mmseg.visualization import SegLocalVisualizer
from .utils import ImageType, _preprare_data


from PIL import Image
import matplotlib.pyplot as plt
import os
from typing import Union, Optional



def init_model(config: Union[str, Path, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None):
    """Initialize a segmentor from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
        cfg_options (dict, optional): Options to override some settings in
            the used config.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if config.model.type == 'EncoderDecoder':
        if 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
    elif config.model.type == 'MultimodalEncoderDecoder':
        for k, v in config.model.items():
            if isinstance(v, dict) and 'init_cfg' in v:
                config.model[k].init_cfg = None
    config.model.pretrained = None
    config.model.train_cfg = None
    init_default_scope(config.get('default_scope', 'mmseg'))

    model = MODELS.build(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        dataset_meta = checkpoint['meta'].get('dataset_meta', None)
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmseg 1.x
            model.dataset_meta = dataset_meta
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmseg 1.x
            classes = checkpoint['meta']['CLASSES']
            palette = checkpoint['meta']['PALETTE']
            model.dataset_meta = {'classes': classes, 'palette': palette}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, classes and palette will be'
                'set according to num_classes ')
            num_classes = model.decode_head.num_classes
            dataset_name = None
            for name in dataset_aliases.keys():
                if len(get_classes(name)) == num_classes:
                    dataset_name = name
                    break
            if dataset_name is None:
                warnings.warn(
                    'No suitable dataset found, use Cityscapes by default')
                dataset_name = 'cityscapes'
            model.dataset_meta = {
                'classes': get_classes(dataset_name),
                'palette': get_palette(dataset_name)
            }
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_model(model: BaseSegmentor,
                    img: ImageType) -> Union[SegDataSample, SampleList]:
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        :obj:`SegDataSample` or list[:obj:`SegDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the segmentation results directly.
    """
    # prepare data
    data, is_batch = _preprare_data(img, model)

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    return results if is_batch else results[0]


def show_result_pyplot(model: BaseSegmentor,
                       img: Union[str, np.ndarray],
                       result: SegDataSample,
                       title: str = '',
                       wait_time: float = 0,
                       show: bool = True,
                       save_dir=None,
                       out_file=None,
                       opacity=0.5,  # Overlay opacity
                       **kwargs):
    """
    Overlay the predicted segmentation mask on the input image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image path or loaded image in BGR order.
        result (SegDataSample): Predicted segmentation result.
        title (str): Display title.
        wait_time (float): Display interval.
        show (bool): Whether to display the overlay.
        save_dir (str, optional): Output directory.
        out_file (str, optional): Output filename.
        opacity (float): Mask opacity between 0.0 and 1.0.
        **kwargs: Additional unused keyword arguments.

    Returns:
        np.ndarray: Image with the predicted mask overlaid, in BGR order.
    """

    # Load the source image.
    if isinstance(img, str):
        # mmcv.imread loads images in BGR order by default.
        img_bgr = mmcv.imread(img)
    elif isinstance(img, np.ndarray):
        img_bgr = img # Expect BGR channel order.
    else:
        raise TypeError("img must be a path string or a NumPy array")

    # Make the image array writable.
    img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)

    # Extract the predicted mask.
    pred_mask_tensor = result.pred_sem_seg.data.squeeze()
    pred_mask_np = pred_mask_tensor.cpu().numpy() # (H, W)

    # Build a BGR mask for fracture pixels.
    # Class 1 is fracture; class 0 is background.
    # Fracture pixels appear white in BGR order.

    # Allocate a three-channel mask.
    color_mask = np.zeros((pred_mask_np.shape[0], pred_mask_np.shape[1], 3), dtype=np.uint8)

    # Display every non-background class in white.
    color_mask[pred_mask_np > 0] = [255, 255, 255] # White in BGR order.

    # Blend the source image and the mask.
    overlay_bgr = cv2.addWeighted(img_bgr, 1 - opacity, color_mask, opacity, 0)

    # Save the BGR overlay if requested.
    if out_file is not None:
        save_path = out_file
        if save_dir is not None:
            mkdir_or_exist(save_dir)
            save_path = os.path.join(save_dir, out_file)

        # Write the BGR image.
        mmcv.imwrite(overlay_bgr, save_path)
        print(f"Overlay mask saved to: {save_path}")

    # Convert to RGB for display.
    if show:
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        plt.imshow(overlay_rgb) # Matplotlib expects RGB.
        plt.title(title)
        plt.axis('off')
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
            plt.close()

    # Return the overlay in BGR order.
    return overlay_bgr
