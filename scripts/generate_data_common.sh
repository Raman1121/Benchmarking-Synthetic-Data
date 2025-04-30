#!/bin/bash

export MODEL_PATH="/pvc/Sana/output/Sana_Diffusers/"
export MODEL_NAME="Sana"
export EXTRA_INFO="Sana"

export REAL_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv"
export SAVE_DIR="/pvc/SYNTHETIC_IMAGES_NEW/"

export SUBSET=300

export BATCH_SIZE=128

python tools/generate_data_common.py \
    --model_path=$MODEL_PATH \
    --model_name=$MODEL_NAME \
    --extra_info=$EXTRA_INFO \
    --real_csv=$REAL_CSV \
    --savedir=$SAVE_DIR \
    --batch_size=$BATCH_SIZE \
    --subset=$SUBSET