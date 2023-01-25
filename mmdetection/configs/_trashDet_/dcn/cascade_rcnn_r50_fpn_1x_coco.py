_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    './schedule_CosineAnnealing.py', '../_base_/default_runtime.py'
]
