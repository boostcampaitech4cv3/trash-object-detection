import cv2
import torch
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

def inference(experience_name, iou=0.5):
    dataset_root = '../dataset/test/'
    
    model = torch.hub.load('./', 'custom', path_or_model= f'./yolov7/{experience_name}/weights/best.pt', source='local') 
    model.conf = 0.001  # confidence threshold (0-1)
    model.iou = iou  # NMS IoU threshold (0-1)

    prediction_string = ['']  * 4871 
    image_id = [f'test/{i:04}.jpg' for i in range(4871)]
    for i in tqdm(range(4871)):
        img = Image.open(os.path.join(dataset_root, f'{i:04}.jpg'))
        
        results = model(img, size=1024, augment=True)
        for bbox in results.pandas().xyxy[0].values:
            xmin, ymin, xmax, ymax, confidence, clss, name = bbox
            prediction_string[i] += f'{clss} {confidence} {xmin} {ymin} {xmax} {ymax} '
    raw_data ={
        'PredictionString' : prediction_string,
        'image_id' : image_id
    }
    dataframe = pd.DataFrame(raw_data)

    # output/yolov7/exp_name에 저장됩니다.
    dataframe.to_csv(f'./yolov7/{experience_name}/submission_{experience_name}_{iou}.csv', sep=',', na_rep='NaN', index=None)
    
if __name__ == '__main__':
    # experiment_name = 'yolov7-e6e_fold_0'
    # inference(experience_name=experiment_name, iou=0.6)
    
    for fold in range(5):
        # experiment_name = f'yolov7-e6e_fold_{fold}-CosineannealingWarmupRestart'
        experiment_name = f'yolov7-e6e_fold_{fold}'
        inference(experience_name=experiment_name, iou=0.4)