_base_ = [
    '../_base_/models/htc_without_semantic_swin_fpn.py',
    '../_base_/datasets/coco_detection_aug_4.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='CBSwinTransformer',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False),
    neck=dict(type='CBFPN', in_channels=[128, 256, 512, 1024]),
    test_cfg=dict(rcnn=dict(score_thr=0.001, nms=dict(type='soft_nms'))))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001 / 2,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

fp16 = dict(loss_scale=dict(init_scale=512))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
# auto_scale_lr = dict(base_batch_size=8*samples_per_gpu)
