#!/bin/bash

export REID_CKPT="/raid/s2198939/diffusion_memorization/PatientVerification/checkpoints/ResNet-50_epoch11_data_handling_RPN.pth"

export REAL_CSV="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/LLAVARAD_ANNOTATIONS_TRAIN.csv"
export REAL_IMG_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export GEN_SAVEDIR="/raid/s2198939/PatientReIdentification/Generations/"

export MODEL_PATH="/raid/s2198939/Sana/output/Sana_Diffusers/epoch_20_step_178057"    # Diffusers pipeline path obtained by using the script "scripts/convert_sana_to_diffusers.sh"
export MODEL_NAME="sana"

export RESULTS_SAVEDIR="Results/"   

export SUBSET=2000

CUDA_VISIBLE_DEVICES=4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python metrics/privacy_metrics.py \
    --model_name=$MODEL_NAME \
    --model_path=$MODEL_PATH \
    --reid_ckpt=$REID_CKPT \
    --real_csv=$REAL_CSV \
    --real_img_dir=$REAL_IMG_DIR \
    --gen_savedir=$GEN_SAVEDIR \
    --results_savedir=$RESULTS_SAVEDIR \
    --subset=$SUBSET

    # --model_path=$MODEL_PATH \
