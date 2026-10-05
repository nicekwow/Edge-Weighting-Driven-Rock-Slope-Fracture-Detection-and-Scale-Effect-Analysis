"""Pass aligned training images to the image-Sobel loss.

These subclasses retain the EncoderDecoder and PSPHead network operations
and parameter names. They change only the training-loss data flow.
Prediction uses the inherited MMSegmentation implementation.
"""

import torch.nn as nn

from mmseg.models.decode_heads.psp_head import PSPHead
from mmseg.models.losses import accuracy
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import add_prefix


@MODELS.register_module(force=True)
class SobelImagePSPHead(PSPHead):
    """PSPHead whose training loss also receives the preprocessed RGB input."""

    def loss(self, inputs, batch_data_samples, train_cfg, image_inputs=None):
        seg_logits = self.forward(inputs)
        seg_label = self._stack_batch_gt(batch_data_samples)
        seg_logits = resize(input=seg_logits, size=seg_label.shape[2:],
                            mode='bilinear', align_corners=self.align_corners)
        seg_weight = self.sampler.sample(seg_logits, seg_label) if self.sampler is not None else None
        seg_label = seg_label.squeeze(1)
        modules = self.loss_decode if isinstance(self.loss_decode, nn.ModuleList) else [self.loss_decode]
        losses = {}
        for loss_module in modules:
            kwargs = dict(weight=seg_weight, ignore_index=self.ignore_index)
            if getattr(loss_module, 'requires_image', False):
                kwargs['image_inputs'] = image_inputs
            value = loss_module(seg_logits, seg_label, **kwargs)
            key = loss_module.loss_name
            losses[key] = losses.get(key, 0) + value
        losses['acc_seg'] = accuracy(seg_logits, seg_label, ignore_index=self.ignore_index)
        return losses


@MODELS.register_module(force=True)
class SobelImageEncoderDecoder(EncoderDecoder):
    """EncoderDecoder with explicit image input for decoder loss weighting."""

    def loss(self, inputs, data_samples):
        features = self.extract_feat(inputs)
        decoded = self.decode_head.loss(features, data_samples, self.train_cfg,
                                        image_inputs=inputs)
        losses = add_prefix(decoded, 'decode')
        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(features, data_samples))
        return losses
