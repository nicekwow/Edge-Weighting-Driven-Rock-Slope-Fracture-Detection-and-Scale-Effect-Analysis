"""Optional mask-Sobel comparison; the manuscript model uses ours.py."""

_base_ = ['./ours.py']

model = dict(type='EncoderDecoder', decode_head=dict(type='PSPHead', loss_decode=[
    dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
    dict(type='SobelMaskEdgeAwareLoss', loss_name='loss_edge', loss_weight=10.0,
         edge_weight=1.0, include_base_loss=False, avg_non_ignore=True),
]))
work_dir = './work_dirs/ours_sobel_mask'
