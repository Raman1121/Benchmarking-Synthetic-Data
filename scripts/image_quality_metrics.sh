#!/bin/bash

export SYNTHETIC_CSV="assets/CSV/prompt_INFO_Sana_ckpt20.csv"
export SYNTHETIC_IMG_DIR="/raid/s2198939/Sana/output/debug/vis/custom_epoch20_step178057_scale4.5_step20_size512_bs1_sampflow_dpm-solver_seed0_float16_flowshift3.0_imgnums100000"

export RESULTS_SAVEDIR="Results"

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export EXTRA_INFO="Sana(ckpt20)"
export EXPERIMENT_TYPE="regular" # "regular" or "conditional" | "regular" calculates the metrics over all pathologies at once

export BATCH_SIZE=64
export NUM_WORKERS=4
export CUDA_VISIBLE_DEVICES=7

echo "Calculating FID, KID, IS ..."
./scripts/fid.sh

echo "Calculating Image Text Alignment Scores ..."
./scripts/img_text_alignment.sh

# Running for CKPT13 Sana
# echo "Running for CKPT13 Sana"
# export EXTRA_INFO="Sana(ckpt13)"
# export SYNTHETIC_CSV="assets/CSV/prompt_INFO_sana_ckpt13.csv"
# export SYNTHETIC_IMG_DIR="/raid/s2198939/Sana/output/debug/vis/custom_epoch13_step103000_scale4.5_step20_size512_bs1_sampflow_dpm-solver_seed0_float16_flowshift3.0_imgnums100000"

# echo "Calculating FID, KID, IS ..."
# ./scripts/fid2.sh

# echo "Calculating Image Text Alignment Scores ..."
# ./scripts/img_text_alignment.sh

# # Running for RadEdit
# echo "Running for RadEdit"
# export EXTRA_INFO="RadEdit"
# export SYNTHETIC_CSV="assets/CSV/promtp_INFO_RadEdit.csv"
# export SYNTHETIC_IMG_DIR="/raid/s2198939/diffusion_memorization/RadEdit_Generations_MIMIC"

# echo "Calculating FID, KID, IS ..."
# ./scripts/fid2.sh

# echo "Calculating Image Text Alignment Scores ..."
# ./scripts/img_text_alignment.sh

# # Running for Pixart
# echo "Running for Pixart"
# export EXTRA_INFO="PixartSigma(ckpt20)"
# export SYNTHETIC_CSV="assets/CSV/prompt_INFO_Pixart_sigma_ckpt20.csv"
# export SYNTHETIC_IMG_DIR="/raid/s2198939/PixArt-sigma/MIMIC_TRAINING1/vis/2025-04-13_custom_epoch20_step51916_scale4.5_step20_size512_bs2_sampdpm-solver_seed0"

# echo "Calculating FID, KID, IS ..."
# ./scripts/fid2.sh

# echo "Calculating Image Text Alignment Scores ..."
# ./scripts/img_text_alignment.sh