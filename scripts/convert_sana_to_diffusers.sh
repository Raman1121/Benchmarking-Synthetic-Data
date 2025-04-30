#!/bin/bash

# source /raid/s2198939/miniconda3/bin/activate sana

CKPT_PATH="/raid/s2198939/Sana/output/debug/checkpoints/epoch_20_step_178057.pth"
SAVE_PATH="/raid/s2198939/Sana/output/Sana_Diffusers/"
IMAGE_SIZE=512
MODEL_TYPE="SanaMS_600M_P1_D28"


python tools/convert_sana_to_diffusers.py \
        --orig_ckpt_path=$CKPT_PATH \
        --image_size=$IMAGE_SIZE \
        --model_type=$MODEL_TYPE \
        --dump_path=$SAVE_PATH \
        --save_full_pipeline
