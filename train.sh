#!/usr/bin/env bash


python train.py



# distributed training

# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1233 train.py --cfg ./configs/coco.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --cfg ./configs/coco.yaml