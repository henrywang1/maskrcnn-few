# #!/bin/bash

export NGPUS=2
echo "Positional Parameters"
# echo '$1 = ' $1
# echo '$2 = ' $2
dst=$1
# ls -Art ~/far_projects_201 | tail -n 1

src="`ls ~/far_projects_201/$1/*.pth -tr | tail -1`"
# src=~/far_projects_201/$1/model_final.pth
~/far_projects_201/$1/model_final.pth
src2=~/far_projects_201/$1/last_checkpoint
split=${2:-1}
way=${3:-0}
shot=${4:-1}   
echo '$src = ' $src
echo '$dst = ' $dst
echo '$split = '$split
echo '$way = '$way
echo '$shot = '$shot
echo "ls ~/far_projects_201/$dst"
ls ~/far_projects_201/$dst
if ! test -d "$dst"; then
    mkdir $dst
fi
cp -u $src "./$dst/"
cp -u $src2 "./$dst/"

python -m torch.distributed.launch --nproc_per_node=$NGPUS \
tools/test_net.py --config-file  "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
TEST.IMS_PER_BATCH $NGPUS OUTPUT_DIR $dst TEST.EXTRACT_FEATURE True DATASETS.SPLIT $split

python -m torch.distributed.launch --nproc_per_node=$NGPUS \
tools/test_net.py --config-file  "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
TEST.IMS_PER_BATCH $NGPUS TEST.USE_FEATURE True \
DATASETS.SPLIT $split OUTPUT_DIR $dst \
TEST.WAY 0 TEST.SHOT $shot \
