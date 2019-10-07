# 說明文件
## Installation
- 參照 [INSTALL.md](INSTALL.md)，Pytorch 1.2 可以正常執行
- 也可使用 [Anaconda Environment](https://gist.github.com/henrywang1/9ab89125d9c4e915d59e7abe0eb99906) 安裝相關套件
- 最好開一個 branch 開發自己的功能

## How to train
### Datasets
參考 [官方文件](https://github.com/henrywang1/maskrcnn-few#perform-training-on-coco-dataset)，但要改成 train2017, val2017

### Arguments
修改 [`train.sh`](train.sh)，訓練參數 (batch size, learning rate, iteration)
  - 1 GPU:
    - SOLVER.IMS_PER_BATCH 1
    - BASE_LR 0.0025
    - MAX_ITER 560000
    - STEPS (480000, )
 - 2 GPUs:
    - SOLVER.IMS_PER_BATCH 2
    - BASE_LR 0.005
    - MAX_ITER 210000
    - STEPS: (240000, )
  - 4 GPUs:
    - SOLVER.IMS_PER_BATCH 4
    - BASE_LR 0.01
    - MAX_ITER 140000
    - STEPS (120000, )
  - 8 GPUs:
    - SOLVER.IMS_PER_BATCH 8
    - BASE_LR 0.02
    - MAX_ITER 70000
    - STEPS (60000, )

## How to test
- 修改 [`test.sh`](test.sh)
- 爲了方便測試，Test 分為兩個步驟，
    1. 把每一張影像的 RoI feature 存起來
    2. 對每一個出現在 query image 上的 class, andom 選另一張影像的 RoI feature 當作 class prototypes

## Limitation
- 目前每顆GPU只能放一張 query image 跟一張 support image (8顆GPU就是 8 query/support)

## ToDo 
- 整合 AAAI 版本其他的功能
    * support more datasets
    * visualization
- CVPR
    * MCG proposal baseline
    * MIL loss
