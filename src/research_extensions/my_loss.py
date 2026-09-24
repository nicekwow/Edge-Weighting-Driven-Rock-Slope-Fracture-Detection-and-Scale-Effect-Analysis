# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.losses.utils import get_class_weight, weight_reduce_loss


def get_categorical_edge_map(target: torch.Tensor,
                             ignore_index: int = -100) -> torch.Tensor:
    """Build an edge map from categorical segmentation labels.

    A pixel is considered an edge when its valid 3x3 neighbourhood contains
    more than one class. Replicate padding prevents the image border from
    being treated as an edge. Neighbourhoods containing ignored pixels are
    excluded so that ignored regions cannot create artificial boundaries.

    Args:
        target (Tensor): Label tensor with shape ``(N, H, W)``.
        ignore_index (int): Label value to ignore. Defaults to -100.

    Returns:
        Tensor: A float edge map with shape ``(N, H, W)`` whose values are 0
        or 1.
    """
    if target.dim() != 3:
        raise ValueError('target must have shape (N, H, W), but got '
                         f'{tuple(target.shape)}')

    valid_mask = target != ignore_index
    labels = target.unsqueeze(1).float()
    valid = valid_mask.unsqueeze(1).float()

    # Replicate padding compares only pixels inside the image and therefore
    # does not introduce an artificial class-0 boundary at the outer border.
    labels = F.pad(labels, (1, 1, 1, 1), mode='replicate')
    valid = F.pad(valid, (1, 1, 1, 1), mode='replicate')

    local_max = F.max_pool2d(labels, kernel_size=3, stride=1)
    local_min = -F.max_pool2d(-labels, kernel_size=3, stride=1)
    all_valid = -F.max_pool2d(-valid, kernel_size=3, stride=1) == 1

    edge_map = (local_max != local_min) & all_valid
    return edge_map.squeeze(1).float()


def edge_aware_loss(pred: torch.Tensor,
                    target: torch.Tensor,
                    weight=None,
                    class_weight=None,
                    reduction='mean',
                    avg_factor=None,
                    ignore_index=-100,
                    avg_non_ignore=True,
                    edge_weight=1.0,
                    include_base_loss=False):
    """Calculate categorical edge-aware cross-entropy loss.

    By default this function returns only the additional boundary term and is
    intended to be combined with a regular cross-entropy loss. Set
    ``include_base_loss=True`` when using it as the sole segmentation loss.
    """
    if pred.dim() != 4:
        raise ValueError('pred must have shape (N, C, H, W), but got '
                         f'{tuple(pred.shape)}')
    if target.dim() != 3:
        raise ValueError('target must have shape (N, H, W), but got '
                         f'{tuple(target.shape)}')
    if pred.shape[0] != target.shape[0] or pred.shape[2:] != target.shape[1:]:
        raise ValueError('pred and target batch/spatial shapes must match, '
                         f'but got {tuple(pred.shape)} and '
                         f'{tuple(target.shape)}')
    if edge_weight < 0:
        raise ValueError('edge_weight must be non-negative')

    valid_mask = target != ignore_index
    pixel_weight = valid_mask.to(dtype=pred.dtype)
    if weight is not None:
        pixel_weight = weight.to(dtype=pred.dtype) * pixel_weight

    basic_loss = F.cross_entropy(
        pred,
        target,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    edge_map = get_categorical_edge_map(target, ignore_index).to(
        dtype=basic_loss.dtype)
    loss_scale = edge_weight * edge_map
    if include_base_loss:
        loss_scale = loss_scale + 1
    loss = basic_loss * loss_scale

    if reduction == 'mean' and avg_factor is None and avg_non_ignore:
        if class_weight is None:
            avg_factor = valid_mask.sum()
        else:
            # Match weighted cross-entropy semantics without indexing the
            # ignore label (which may lie outside the class range).
            safe_target = target.masked_fill(~valid_mask, 0)
            avg_factor = (class_weight[safe_target] * valid_mask).sum()

    return weight_reduce_loss(
        loss, pixel_weight, reduction=reduction, avg_factor=avg_factor)


@MODELS.register_module(force=True)
class EdgeAwareLoss(nn.Module):
    """Categorical edge-aware loss for semantic segmentation.

    Args:
        reduction (str): Reduction method: ``none``, ``mean`` or ``sum``.
            Defaults to ``mean``.
        class_weight (list[float] | str, optional): Weight of each class.
        loss_weight (float): Overall loss multiplier. Defaults to 1.0.
        loss_name (str): Name used in the MMSeg loss dictionary. Defaults to
            ``loss_edge``.
        avg_non_ignore (bool): Average over non-ignored targets. Defaults to
            True.
        edge_weight (float): Strength of the additional boundary term.
            Defaults to 1.0.
        include_base_loss (bool): Include ordinary cross-entropy at every
            valid pixel. Keep this False when a separate CrossEntropyLoss is
            configured, and set it True when this is the sole loss. Defaults
            to False.
    """

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_edge',
                 avg_non_ignore=True,
                 edge_weight=1.0,
                 include_base_loss=False):
        super().__init__()
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f'Unsupported reduction: {reduction}')
        if edge_weight < 0:
            raise ValueError('edge_weight must be non-negative')

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        self.edge_weight = edge_weight
        self.include_base_loss = include_base_loss
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        if reduction_override not in (None, 'none', 'mean', 'sum'):
            raise ValueError(
                f'Unsupported reduction_override: {reduction_override}')
        reduction = reduction_override or self.reduction
        class_weight = (pred.new_tensor(self.class_weight)
                        if self.class_weight is not None else None)

        return self.loss_weight * edge_aware_loss(
            pred,
            target,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            edge_weight=self.edge_weight,
            include_base_loss=self.include_base_loss)

    @property
    def loss_name(self):
        """Return the loss name used by MMSeg to collect train losses."""
        return self._loss_name
