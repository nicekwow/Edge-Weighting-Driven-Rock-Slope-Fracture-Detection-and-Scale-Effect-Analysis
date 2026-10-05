"""Sobel edge-weighted CE variants for a controlled training comparison.

Both variants use the same Sobel kernels and fixed magnitude normalization.
SobelMaskEdgeAwareLoss obtains weights from binary reference masks;
SobelImageEdgeAwareLoss obtains them from the aligned training RGB images.
They supply an additional term alongside ordinary cross-entropy.
"""

import math

import torch
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.losses.utils import weight_reduce_loss
from .my_loss import EdgeAwareLoss


def sobel_magnitude(gray):
    """Return fixed-scale Sobel magnitudes for (N, 1, H, W) values in [0, 1].

    sqrt(20) is the maximum magnitude of these unscaled 3x3 Sobel kernels
    for samples in [0, 1]. Fixed normalization preserves relative contrast
    across images; it does not amplify each image to its own maximum.
    """
    if gray.ndim != 4 or gray.shape[1] != 1:
        raise ValueError('gray must have shape (N, 1, H, W)')
    kernels = gray.new_tensor([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    ]).unsqueeze(1)
    gradients = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='replicate'), kernels)
    return gradients.square().sum(dim=1).sqrt() / math.sqrt(20)


def valid_neighbourhood(target, ignore_index):
    """Exclude padded/ignored pixels and their 3x3 neighbourhoods."""
    valid = (target != ignore_index).unsqueeze(1).float()
    valid = F.pad(valid, (1, 1, 1, 1), mode='replicate')
    return (-F.max_pool2d(-valid, kernel_size=3, stride=1) == 1).squeeze(1)


def get_sobel_mask_edge_map(target, ignore_index=-100):
    """Compute Sobel weights from binary labels, with ignored areas excluded."""
    labels = target.masked_fill(target == ignore_index, 0).unsqueeze(1).float()
    return sobel_magnitude(labels) * valid_neighbourhood(target, ignore_index)


class _SobelEdgeAwareLoss(EdgeAwareLoss):
    """Shared CE weighting/reduction; subclasses supply the edge map."""

    requires_image = False

    def edge_map(self, target, ignore_index, image_inputs):
        raise NotImplementedError

    def forward(self, pred, target, weight=None, avg_factor=None,
                reduction_override=None, ignore_index=-100, image_inputs=None,
                **kwargs):
        if pred.ndim != 4 or target.ndim != 3 or pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
            raise ValueError('pred (N, C, H, W) and target (N, H, W) must align')
        if reduction_override not in (None, 'none', 'mean', 'sum'):
            raise ValueError('unsupported reduction_override')
        reduction = reduction_override or self.reduction
        class_weight = pred.new_tensor(self.class_weight) if self.class_weight is not None else None
        valid = target != ignore_index
        pixel_weight = valid.to(pred.dtype)
        if weight is not None:
            pixel_weight = pixel_weight * weight.to(pred.dtype)
        pixel_ce = F.cross_entropy(pred, target, weight=class_weight,
                                   reduction='none', ignore_index=ignore_index)
        edges = self.edge_map(target, ignore_index, image_inputs).to(pixel_ce.dtype)
        scale = self.edge_weight * edges
        if self.include_base_loss:
            scale = scale + 1
        if reduction == 'mean' and avg_factor is None and self.avg_non_ignore:
            if class_weight is None:
                avg_factor = valid.sum()
            else:
                safe_target = target.masked_fill(~valid, 0)
                avg_factor = (class_weight[safe_target] * valid).sum()
        return self.loss_weight * weight_reduce_loss(
            pixel_ce * scale, pixel_weight, reduction=reduction, avg_factor=avg_factor)


@MODELS.register_module(force=True)
class SobelMaskEdgeAwareLoss(_SobelEdgeAwareLoss):
    """Weight CE with the Sobel magnitude of the binary reference mask."""

    def edge_map(self, target, ignore_index, image_inputs):
        return get_sobel_mask_edge_map(target, ignore_index)


@MODELS.register_module(force=True)
class SobelImageEdgeAwareLoss(_SobelEdgeAwareLoss):
    """Weight CE with Sobel gradients of aligned, augmented RGB images.

    Inputs arrive after SegDataPreProcessor normalization. Its mean and std
    are reversed before grayscale conversion; gradients are therefore
    computed on image intensities rather than normalized network features.
    The image weights are detached from the model gradient graph.
    """

    requires_image = True

    def __init__(self, mean=(123.675, 116.28, 103.53),
                 std=(58.395, 57.12, 57.375), **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('image_mean', torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('image_std', torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def edge_map(self, target, ignore_index, image_inputs):
        if image_inputs is None:
            raise ValueError('SobelImageEdgeAwareLoss requires aligned image_inputs')
        if image_inputs.ndim != 4 or image_inputs.shape[1] != 3 or image_inputs.shape[0] != target.shape[0] or image_inputs.shape[2:] != target.shape[1:]:
            raise ValueError('image_inputs (N, 3, H, W) must align with target')
        rgb = (image_inputs.detach().float() * self.image_std + self.image_mean).clamp(0, 255) / 255
        coefficients = rgb.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        gray = (rgb * coefficients).sum(dim=1, keepdim=True)
        return sobel_magnitude(gray) * valid_neighbourhood(target, ignore_index)
