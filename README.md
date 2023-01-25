# CV-16조 비전길잡이

![image](https://user-images.githubusercontent.com/25689849/214464593-4d772adc-3bc4-408a-82c6-0ba21a6b8d1c.png)

![result]()


## Contributors
|민기|박민지|유영준|장지훈|최동혁|
|:----:|:----:|:----:|:---:|:---:|
|[<img alt="revanZX" src="https://avatars.githubusercontent.com/u/25689849?v=4&s=100" width="100">](https://github.com/revanZX)|[<img alt="arislid" src="https://avatars.githubusercontent.com/u/46767966?v=4&s=100" width="100">](https://github.com/arislid)|[<img alt="youngjun04" src="https://avatars.githubusercontent.com/u/113173095?v=4&s=100" width="100">](https://github.com/youngjun04)|[<img alt="FIN443" src="https://avatars.githubusercontent.com/u/70796031?v=4&s=100" width="100">](https://github.com/FIN443)|[<img alt="choipp" src="https://avatars.githubusercontent.com/u/103131249?v=4&s=117" width="100">](https://github.com/choipp)|


## Directory Structure
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

## Appendix
[CV16조 발표자료 & WarpUpReport](./appendix/)