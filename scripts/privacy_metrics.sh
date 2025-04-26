#!/bin/bash

export REID_CKPT="/pvc/PatientReIdentification/ResNet-50_epoch11_data_handling_RPN.pth"

export REAL_CSV="/pvc/MIMIC_SPLITS/LLAVARAD_ANNOTATIONS_TRAIN.csv"
export REAL_IMG_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export GEN_SAVEDIR="/pvc/PatientReIdentification/Generations/"

export MODEL_PATH="/pvc/Benchmarking-Synthetic-Data/OUTPUT_MIMIC_SD_V1_4/IID/512/SD-V1-4_IID_512"
export MODEL_NAME="SD-V1-4"

export RESULTS_SAVEDIR="/data/Benchmarking-Synthetic-Data/Results/"

export SUBSET=20

python metrics/privacy_metrics.py \
    --model_name=$MODEL_NAME \
    --model_path=$MODEL_PATH \
    --subset=$SUBSET \
    --reid_ckpt=$REID_CKPT \
    --real_csv=$REAL_CSV \
    --real_img_dir=$REAL_IMG_DIR \
    --gen_savedir=$GEN_SAVEDIR \
    --results_savedir=$RESULTS_SAVEDIR
