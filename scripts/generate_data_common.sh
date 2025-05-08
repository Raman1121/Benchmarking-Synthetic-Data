#!/bin/bash

export MODEL_PATH="/raid/s2198939/medical-diffusion-classifier/OUTPUT_MIMIC_SD_V1_5_Impressions/SD-V1-5_IID_512"
export MODEL_NAME="SD-V1-5"
export EXTRA_INFO="SD-V1-5-Impressions"

export REAL_CSV="/raid/s2198939/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv"
export SAVE_DIR="/raid/s2198939/SYNTHETIC_IMAGES_NEW/"

# export SUBSET=200

export BATCH_SIZE=32

CUDA_VISIBLE_DEVICES=4 python tools/generate_data_common.py \
    --model_path=$MODEL_PATH \
    --model_name=$MODEL_NAME \
    --extra_info=$EXTRA_INFO \
    --real_csv=$REAL_CSV \
    --savedir=$SAVE_DIR \
    --batch_size=$BATCH_SIZE \
    # --subset=$SUBSET