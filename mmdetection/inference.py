import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
from multiprocessing import freeze_support

def main(k_i, b_i):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
    model_path = "dyhead"
    model_name = "atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco_aug_3"

    k_fold = k_i
    score_thr = 0.001
    root='../../dataset/'
    epoch = f'best_bbox_mAP_50_epoch_{b_i}'

    Vaild = True
    Test = True


    cfg = None
    if Vaild:
        # Get Validation
        # dataset config 수정
        cfg = Config.fromfile(f'./configs/_trashDet_/{model_path}/{model_name}.py')
        cfg.data.test.classes = classes
        cfg.data.test.img_prefix = root
        cfg.data.test.ann_file = root + f'val_{k_fold}.json'
        cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
        cfg.data.test.test_mode = True

        cfg.data.samples_per_gpu = 4

        cfg.seed=24
        cfg.gpu_ids = [1]
        cfg.work_dir = f'./work_dirs/{model_name}_{k_fold}'

        cfg.model.train_cfg = None

        # build dataset & dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
                dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False)

        # checkpoint path
        checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

        model.CLASSES = dataset.CLASSES
        model = MMDataParallel(model.cuda(), device_ids=[0])

        output = single_gpu_test(model, data_loader, show_score_thr=score_thr) # output 계산

        # submission 양식에 맞게 output 후처리
        prediction_strings = []
        file_names = []
        coco = COCO(cfg.data.test.ann_file)
        img_ids = coco.getImgIds()

        class_num = 10
        for i, out in zip(img_ids, output):
            prediction_string = ''
            image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
            for j in range(class_num):
                for o in out[j]:
                    prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                        o[2]) + ' ' + str(o[3]) + ' '

            prediction_strings.append(prediction_string)
            file_names.append(image_info['file_name'])


        submission = pd.DataFrame()
        submission['PredictionString'] = prediction_strings
        submission['image_id'] = file_names
        submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}_{score_thr}_val_{k_fold}.csv'), index=None)
        submission.head()

    if Test:
        # Test
        # dataset config 수정
        cfg = None
        cfg = Config.fromfile(f'./configs/_trashDet_/{model_path}/{model_name}.py')
        cfg.data.test.classes = classes
        cfg.data.test.img_prefix = root
        cfg.data.test.ann_file = root + 'test.json'
        cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
        cfg.data.test.test_mode = True

        cfg.data.samples_per_gpu = 4

        cfg.seed=24
        cfg.gpu_ids = [1]
        cfg.work_dir = f'./work_dirs/{model_name}_{k_fold}'

        cfg.model.train_cfg = None

        # build dataset & dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
                dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False)

        # checkpoint path
        checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

        model.CLASSES = dataset.CLASSES
        model = MMDataParallel(model.cuda(), device_ids=[0])

        output = single_gpu_test(model, data_loader, show_score_thr=score_thr) # output 계산

        # submission 양식에 맞게 output 후처리
        prediction_strings = []
        file_names = []
        coco = COCO(cfg.data.test.ann_file)
        img_ids = coco.getImgIds()

        class_num = 10
        for i, out in zip(img_ids, output):
            prediction_string = ''
            image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
            for j in range(class_num):
                for o in out[j]:
                    prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                        o[2]) + ' ' + str(o[3]) + ' '

            prediction_strings.append(prediction_string)
            file_names.append(image_info['file_name'])


        submission = pd.DataFrame()
        submission['PredictionString'] = prediction_strings
        submission['image_id'] = file_names
        submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}_{score_thr}.csv'), index=None)
        submission.head()
        
if __name__ == '__main__':
    freeze_support()
    main(1,7)
    #for k_i, b_i in zip([0,1,2,3,4],[11,13,13,12,10]):
    #    main(k_i, b_i)