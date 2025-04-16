#!/bin/bash

export SYNTHETIC_CSV="assets/CSV/prompt_INFO_Sana_ckpt20.csv"
export SYNTHETIC_IMG_DIR="/raid/s2198939/Sana/output/debug/vis/custom_epoch20_step178057_scale4.5_step20_size512_bs1_sampflow_dpm-solver_seed0_float16_flowshift3.0_imgnums100000"

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export EXTRA_INFO="Sana(ckpt20)"
export EXPERIMENT_TYPE="regular" # "regular" or "conditional" | "regular" calculates the metrics over all pathologies at once

export RESULTS_SAVEDIR="Results/"
export SHARDS_DIR="Results/saved_shards"
mkdir -p  $SHARDS_DIR

export NUM_SHARDS=4
export SHARD=0

export BATCH_SIZE=64
export NUM_WORKERS=4
export CUDA_VISIBLE_DEVICES=7

# for (( shard=0; shard<NUM_SHARDS; shard++ )); do
#     export SHARD=$shard
#     echo "Calculating FID, KID, IS for shard $SHARD ..."
#     ./scripts/fid.sh
# done

# Combine the shards here
python tools/combine_shards.py --shards_dir=$SHARDS_DIR --extra_info=$EXTRA_INFO --output_dir=$RESULTS_SAVEDIR --delete_after_combining 

# Calculate img-text alignment scores
# echo "Calculating Image Text Alignment Scores ..."
# ./scripts/img_text_alignment.sh

