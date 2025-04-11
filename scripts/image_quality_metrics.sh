#!/bin/bash

export SYNTHETIC_CSV="assets/CSV/promtp_INFO_RadEdit.csv"
export SYNTHETIC_IMG_DIR="/raid/s2198939/diffusion_memorization/RadEdit_Generations_MIMIC"

export RESULTS_SAVEDIR="Results"

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export EXTRA_INFO="RadEdit"

export BATCH_SIZE=64
export NUM_WORKERS=4
export CUDA_VISIBLE_DEVICES=2

echo "Calculating FID, KID, IS ..."
./scripts/fid.sh

echo "Calculating Image Text Alignment Scores ..."
./scripts/img_text_alignment.sh