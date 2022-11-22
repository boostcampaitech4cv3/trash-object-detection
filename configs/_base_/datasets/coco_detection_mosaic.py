# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10, 
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict( 
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
                    dict(
                        type='Resize',
                        img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                    (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                    (736, 1333), (768, 1333), (800, 1333)],
                        multiscale_mode='value',
                        keep_ratio=True)
                  ],
                  [
                    dict(
                        type='Resize',
                        img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(
                        type='RandomCrop',
                        crop_type='absolute_range',
                        crop_size=(384, 600),
                        allow_negative_crop=True),
                    dict(
                        type='Resize',
                        img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                        multiscale_mode='value',
                        override=True,
                        keep_ratio=True),
                    # dict(
                    #     type='PhotoMetricDistortion',
                    #     brightness_delta=32,
                    #     contrast_range=(0.5, 1.5),
                    #     saturation_range=(0.5, 1.5),
                    #     hue_delta=18),
                    # dict(
                    #     type='MinIoURandomCrop',
                    #     min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                    #     min_crop_size=0.3),
                ],
                [
                    dict(
                        type='Resize',
                        img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(
                        type='Mosaic',
                        img_scale=(640, 640),
                        center_ratio_range=(0.5, 1.5),
                        min_bbox_size=0,
                        bbox_clip_border=True,
                        skip_filter=True,
                        pad_val=114,
                        prob=1.0),
                    dict(
                        type='Resize',
                        img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                        multiscale_mode='value',
                        override=True,
                        keep_ratio=True),
                ],
                [
                    dict(
                        type='Resize',
                        img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                    (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                    (736, 1333), (768, 1333), (800, 1333)],
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
                            'gt_masks': 'masks',
                            'gt_bboxes': 'bboxes'
                        },
                        update_pad_shape=False,
                        skip_img_without_anno=True),
                ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
