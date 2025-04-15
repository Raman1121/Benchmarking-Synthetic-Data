#!/bin/bash

export SYNTHETIC_CSV="assets/CSV/prompts_INFO_sana_ckpt19.csv"
export SYNTHETIC_IMG_DIR="/raid/s2198939/Sana/output/debug/vis/custom_epoch19_step170638_scale4.5_step20_size512_bs16_sampflow_dpm-solver_seed0_float16_flowshift3.0_imgnums100000"

export RESULTS_SAVEDIR="Results"

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export EXTRA_INFO="Sana(ckpt19)"

export BATCH_SIZE=128
export NUM_WORKERS=4
export CUDA_VISIBLE_DEVICES=7

MIMIC_PATHOLOGIES=("Atelectasis" "Cardiomegaly" "Consolidation" "Edema" "Enlarged Cardiomediastinum" "Fracture" "Lung Lesion" "Lung Opacity" "No Finding" "Pleural Effusion" "Pleural Other" "Pneumonia" "Pneumothorax" "Support Devices")

for pathology in "${MIMIC_PATHOLOGIES[@]}"; do
    export EXPERIMENT_TYPE="conditional"
    export PATHOLOGY=$pathology # Specify this if experiment type is "conditional"
    echo "Conditional Experiment for: '$PATHOLOGY'"
    echo "Calculating FID, KID, IS ..."
    ./scripts/fid.sh

    echo "Calculating Image Text Alignment Scores ..."
    ./scripts/img_text_alignment.sh
done
