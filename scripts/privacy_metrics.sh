#!/bin/bash

export REID_CKPT="/pvc/PatientReIdentification/ResNet-50_epoch11_data_handling_RPN.pth"

export REAL_CSV="/pvc/MIMIC_SPLITS/LLAVARAD_ANNOTATIONS_TRAIN.csv"
export REAL_IMG_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export GEN_SAVEDIR="/pvc/PatientReIdentification/Generations/"

export MODEL_PATH="/pvc/ai-toolkit/output/flux_lora"
export MODEL_NAME="flux"
# export EXTRA_INFO=
# export MODEL_PATH="/pvc/ai-toolkit/output/lumina2_lora"
# export MODEL_NAME="lumina"

export RESULTS_SAVEDIR="/pvc/Benchmarking-Synthetic-Data/Results/"
export SUBSET=2000

python metrics/privacy_metrics.py \
    --model_name=$MODEL_NAME \
    --model_path=$MODEL_PATH \
    --reid_ckpt=$REID_CKPT \
    --real_csv=$REAL_CSV \
    --real_img_dir=$REAL_IMG_DIR \
    --gen_savedir=$GEN_SAVEDIR \
    --results_savedir=$RESULTS_SAVEDIR \
    --subset=$SUBSET \
    # --extra_info=$EXTRA_INFO
