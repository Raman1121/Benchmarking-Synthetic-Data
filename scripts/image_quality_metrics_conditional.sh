#!/bin/bash

source /raid/s2198939/miniconda3/bin/activate sana

export SYNTHETIC_CSV="assets/CSV/prompt_INFO_Pixart_sigma_ckpt20.csv"
export SYNTHETIC_IMG_DIR="/raid/s2198939/PixArt-sigma/MIMIC_TRAINING1/vis/2025-04-13_custom_epoch20_step51916_scale4.5_step20_size512_bs2_sampdpm-solver_seed0"

export RESULTS_SAVEDIR="Results"

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export EXTRA_INFO="Pixart(ckpt20)"
export EXPERIMENT_TYPE="conditional"

export NUM_SHARDS=-1
export SHARD=-1

export BATCH_SIZE=128
export NUM_WORKERS=4
export CUDA_VISIBLE_DEVICES=7

MIMIC_PATHOLOGIES=("Atelectasis" "Cardiomegaly" "Consolidation" "Edema" "Enlarged Cardiomediastinum" "Fracture" "Lung Lesion" "Lung Opacity" "No Finding" "Pleural Effusion" "Pleural Other" "Pneumonia" "Pneumothorax" "Support Devices")
# MIMIC_PATHOLOGIES=("Pleural Effusion")

for pathology in "${MIMIC_PATHOLOGIES[@]}"; do
    export EXPERIMENT_TYPE="conditional"
    export PATHOLOGY=$pathology # Specify this if experiment type is "conditional"
    echo "Conditional Experiment for: '$PATHOLOGY'"
    echo "Calculating FID, KID, IS ..."
    ./scripts/fid.sh

    echo "Calculating Image Text Alignment Scores ..."
    ./scripts/img_text_alignment.sh
done