#!/bin/bash

source /raid/s2198939/miniconda3/bin/activate sana

SYNTHETIC_CSV=""
SYNTHETIC_IMG_DIR=""

REAL_CSV=""
REAL_IMG_DIR=""

RESULTS_SAVEDIR="results"

BATCH_SIZE=32

python metrics/fid.py --synthetic_csv=$SYNTHETIC_CSV \
                        --synthetic_img_dir=$SYNTHETIC_IMG_DIR \
                        --real_csv=$REAL_CSV \
                        --real_img_dir=$REAL_IMG_DIR \
                        --results_savedir=$RESULTS_SAVEDIR \
                        --batch_size=$BATCH_SIZE
