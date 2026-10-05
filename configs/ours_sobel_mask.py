"""Mask-Sobel variant; retain all other recorded settings from ours.py."""

_base_ = ['./ours.py']

randomness = dict(seed=1498288285, diff_rank_seed=False, deterministic=False)

model = dict(decode_head=dict(loss_decode=[
    dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
    dict(type='SobelMaskEdgeAwareLoss', loss_name='loss_edge', loss_weight=10.0,
         edge_weight=1.0, include_base_loss=False, avg_non_ignore=True),
]))
