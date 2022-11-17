# 모듈 import
import platform

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)
from mmdet.utils import get_device

from multiprocessing import freeze_support

import torch
from mmcv.runner.hooks import HOOKS, Hook

selfos = platform.system() 
work_dir = './work_dirs/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco'

@HOOKS.register_module()
class CheckBestLossHook(Hook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=50):
        self.interval = interval
        self.loss = 10000

    def after_train_epoch(self, runner):
        if runner.outputs['loss'] < self.loss:
            self.loss = runner.outputs['loss']
            runner.logger.info('Best loss found')
            runner.save_checkpoint(out_dir=work_dir, filename_tmpl='best.pth', save_optimizer=False)

def main():
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    cfg = Config.fromfile('./configs/_trashDet_/swin/faster_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py')

    root='../../dataset/'
    # dataset config 수정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train.json' # train json 정보

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보

    cfg.data.samples_per_gpu = 4
    cfg.data.workers_per_gpu = 4

    cfg.seed = 24
    cfg.gpu_ids = [0]
    cfg.work_dir = work_dir

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(interval=1)
    cfg.log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='CheckBestLossHook')
            # dict(type='TensorboardLoggerHook')
        ])
    cfg.device = get_device()

    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    print(datasets[0])

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    meta = dict()
    meta['fp16'] = dict(loss_scale=dict(init_scale=512))

    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=False, meta=meta)

if __name__ == '__main__':
    if selfos == 'Windows':
        freeze_support()
    main()