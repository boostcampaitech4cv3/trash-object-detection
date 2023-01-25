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

model_dir = 'dyhead'
model_name = 'atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_aug_3'
work_dir = f'./work_dirs/{model_name}'

@HOOKS.register_module()
class ImageDetection(Hook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=50):
        self.interval = interval
        
    def after_train_epoch(self, runner):
        print(dir(runner))

def main(k_fold):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    cfg = Config.fromfile(f'./configs/_trashDet_/{model_dir}/{model_name}.py')

    root='../../dataset/'
    # dataset config 수정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + f'train_{k_fold}.json' # train json 정보
    
    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + f'val_{k_fold}.json'
   
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    
    cfg.train_pipeline = cfg.train_pipeline
    cfg.val_pipeline = cfg.test_pipeline
    cfg.test_pipeline = cfg.test_pipeline

    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.val.pipeline = cfg.val_pipeline
    cfg.data.test.pipeline = cfg.test_pipeline

    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 4

    cfg.seed = 24
    cfg.gpu_ids = [0]
    cfg.work_dir = work_dir + f'_{k_fold}'

    cfg.evaluation = dict(
        interval=1, 
        start=1,
        save_best='auto' 
    )
    
    
    cfg.optimizer_config.grad_clip = None #dict(max_norm=35, norm_type=2)
    cfg.optimizer.lr=0.000005
    cfg.lr_config.step=[4]
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            #dict(type='ImageDetection'),
            dict(type='TensorboardLoggerHook')
        ])
    cfg.device = get_device()
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=20)
    cfg.load_from = './work_dirs/dyhead/best_bbox_mAP_50_epoch_12.pth'
    # build_dataset
    datasets = [build_dataset(cfg.data.train)]
    
    #print(datasets[0])

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    meta = dict()
    #meta['fp16'] = dict(loss_scale=dict(init_scale=512))

    # 모델 학습
    train_detector(model, datasets, cfg, distributed=False, validate=True, meta=meta)

if __name__ == '__main__':
    if selfos == 'Windows':
        freeze_support()
    
    main(1)
    #for k_fold in range(5):
    #    main(k_fold)