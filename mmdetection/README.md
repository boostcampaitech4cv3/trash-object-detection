# **🚮 MMdetection Directory Structure**
```
|-- appendix : 이 프로젝트에 대한 발표자료와 WrapUp Report
|-- README.md
|-- configs
|   |-- _trashDet_ : 모든 모델 설정은 _trashDet_을 사용
|   |   |-- _base_
|   |   |   |-- datasets
|   |   |   |-- default_runtime.py
|   |   |   |-- models
|   |   |   `-- schedules
|   |   |-- cbnet
|   |   |-- convnext
|   |   |-- dcn
|   |   |-- detectors
|   |   |-- detr
|   |   |-- dyhead
|   |   |-- efficientnet
|   |   |-- focalnet
|   |   |-- htc
|   |   |-- retinanet
|   |   |-- swin
|   |   |-- swinv2
|   |   |-- timm
|   |   |-- universenet
|   |   `-- yolox
|-- custom_tool  : confusion metrix나 kfold등 자체 제작 툴
|-- dataset      : Kfold 적용
|-- mmcv_custom  : apex 모듈을 사용한 모델 의존 파일
|-- mmdet        : ConvNeXt 등 외부 모델 구현
|-- wandb        : wandb 파일 저장
|-- work_dirs    : 학습 결과물
|-- requirements.txt
|-- train.py     : 동일한 학습을 하기 위한 팀 공용 학습 파일
`-- inference.py : 동일한 추론을 하기 위한 팀 공용 추론 파일
```