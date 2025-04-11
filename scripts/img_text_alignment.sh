#!/bin/bash

source /raid/s2198939/miniconda3/bin/activate himl

echo "SYNTHETIC CSV CSV: $SYNTHETIC_CSV"
echo "SYNTHETIC IMG DIR: $SYNTHETIC_IMG_DIR"
echo "Saving Results at: $RESULTS_SAVEDIR"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python metrics/img_text_alignment_scores.py \
    --synthetic_csv=$SYNTHETIC_CSV \
    --synthetic_img_dir=$SYNTHETIC_IMG_DIR \
    --results_savedir=$RESULTS_SAVEDIR \
    --batch_size=$BATCH_SIZE \
    --num_workers=$NUM_WORKERS \
    --extra_info=$EXTRA_INFO