# Split 1~4 Train: 4i+1, Test: Others classes
# Split 5: Train: COCO, Test: Non-COCO
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
tools/train_net.py --config-file  "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
OUTPUT_DIR models/coco_1 DATASETS.SPLIT 1