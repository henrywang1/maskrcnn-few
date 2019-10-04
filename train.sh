# DATASETS.SPLIT 1~4 Train: 4i+1, Test: Other classes
# DATASETS.SPLIT 5: Train: COCO-Non-Voc, Test: PASCAL VOC
export NGPUS=8

python -m torch.distributed.launch --nproc_per_node=$NGPUS \
tools/train_net.py --config-file  "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
OUTPUT_DIR models/coco_1 DATASETS.SPLIT 1 SOLVER.IMS_PER_BATCH $NGPUS \
SOLVER.BASE_LR 0.02 SOLVER.MAX_ITER 70000 SOLVER.STEPS "(60000,)" 
