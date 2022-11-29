_base_ = [
    '../_base_/models/htc_without_semantic_swin_fpn.py',
    '../_base_/datasets/coco_detection_aug_3.py',
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

dataset_type = 'CocoDataset'
data_root = '../../../dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from HTC
albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='Flip',p=1.0),
            dict(type='RandomRotate90',p=1.0)
        ],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=(-0.1, 0.15),
                contrast_limit=(-0.1, 0.15),
                p=1.0),
            dict(
                type='CLAHE',
                clip_limit=(2, 6),
                tile_grid_size=(8, 8),
                p=1.0),
        ],
        p=0.5),
    dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
    dict(type='GaussNoise', var_limit=(20, 100), p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0),
            dict(type='MotionBlur', p=1.0)
        ],
        p=0.1),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False, with_seg=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize', 
        img_scale=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
                   (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                   (736, 1024), (768, 1024), (800, 1024)],
        multiscale_mode='value', 
        keep_ratio=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True
        ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=True, # <-- True=TTA
        flip_direction=['horizontal', 'vertical', 'diagonal'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
samples_per_gpu = 2
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
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
