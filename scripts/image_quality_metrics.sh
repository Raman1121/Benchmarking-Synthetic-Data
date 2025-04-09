#!/bin/bash

export SYNTHETIC_CSV="assets/CSV/prompts_INFO_epoch10.csv"
export SYNTHETIC_IMG_DIR="/raid/s2198939/Sana/output/debug/vis/custom_epoch10_step75000_scale4.5_step20_size512_bs1_sampflow_dpm-solver_seed0_float16_flowshift3.0_imgnums100000"

export RESULTS_SAVEDIR="Results"

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export BATCH_SIZE=64
export NUM_WORKERS=4
export CUDA_VISIBLE_DEVICES=0

echo "Calculating FID, KID, IS ..."
./scripts/fid.sh

echo "Calculating Image Text Alignment Scores ..."
./scripts/img_text_alignment.sh