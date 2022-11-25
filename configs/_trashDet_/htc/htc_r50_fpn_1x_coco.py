_base_ = './htc_without_semantic_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=183,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2))))

# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../../dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
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
data = dict(
    samples_per_gpu=4,
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
evaluation = dict(interval=1, metric='bbox')