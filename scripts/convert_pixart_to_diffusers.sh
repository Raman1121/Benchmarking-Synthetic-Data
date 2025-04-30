#!/bin/bash

# source /raid/s2198939/miniconda3/bin/activate sana

CKPT_PATH="/raid/s2198939/PixArt-sigma/MIMIC_TRAINING1/checkpoints/epoch_20_step_51916.pth"
SAVE_PATH="/raid/s2198939/PixArt-sigma/MIMIC_TRAINING1/Pixart_diffusers/"
IMAGE_SIZE=512


python tools/convert_pixart_to_diffusers.py \
        --orig_ckpt_path=$CKPT_PATH \
        --image_size=$IMAGE_SIZE \
        --dump_path=$SAVE_PATH \
        --version="sigma" \