"""Image-Sobel variant; retain recorded settings and network parameters."""

_base_ = ['./ours.py']

randomness = dict(seed=1498288285, diff_rank_seed=False, deterministic=False)

model = dict(
    type='SobelImageEncoderDecoder',
    decode_head=dict(type='SobelImagePSPHead', loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        dict(type='SobelImageEdgeAwareLoss', loss_name='loss_edge', loss_weight=10.0,
             edge_weight=1.0, include_base_loss=False, avg_non_ignore=True,
             mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
    ]))
