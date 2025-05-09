#!/bin/bash

#################### Sana ####################
export MODEL_PATH="/pvc/Benchmarking-Synthetic-Data/Sana_Epoch50"
export MODEL_NAME="sana"
export EXTRA_INFO="sana_epoch50"

export REAL_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TRAIN.csv"
export SAVE_DIR="/pvc/SynthCheX/"

export BATCH_SIZE=256

python tools/generate_data_common.py \
    --model_path=$MODEL_PATH \
    --model_name=$MODEL_NAME \
    --extra_info=$EXTRA_INFO \
    --real_csv=$REAL_CSV \
    --savedir=$SAVE_DIR \
    --batch_size=$BATCH_SIZE \
    --use_dicom_id