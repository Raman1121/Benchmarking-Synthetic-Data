#!/bin/bash

source /raid/s2198939/miniconda3/bin/activate sana

SYNTHETIC_CSV="assets/CSV/prompts_INFO.csv"
SYNTHETIC_IMG_DIR="/raid/s2198939/Sana/output/debug/vis/custom_epoch9_step60000_scale4.5_step20_size512_bs1_sampflow_dpm-solver_seed0_float16_flowshift3.0_imgnums100000"
# SYNTHETIC_IMG_DIR="assets/synthetic_images/"

REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
REAL_IMG_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

RESULTS_SAVEDIR="Results"

BATCH_SIZE=32
NUM_WORKERS=4
CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python metrics/fid.py --synthetic_csv=$SYNTHETIC_CSV \
                                                    --synthetic_img_dir=$SYNTHETIC_IMG_DIR \
                                                    --real_csv=$REAL_CSV \
                                                    --real_img_dir=$REAL_IMG_DIR \
                                                    --results_savedir=$RESULTS_SAVEDIR \
                                                    --batch_size=$BATCH_SIZE \
                                                    --num_workers=$NUM_WORKERS \
