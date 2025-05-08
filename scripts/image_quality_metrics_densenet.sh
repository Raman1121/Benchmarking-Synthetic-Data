#!/bin/bash

source /raid/s2198939/miniconda3/bin/activate sana2

export SYNTHETIC_CSV="/raid/s2198939/medical-diffusion-classifier/sd3-5_medium_lora/generated_images/prompt_INFO.csv"
export SYNTHETIC_IMG_DIR="/raid/s2198939/medical-diffusion-classifier/sd3-5_medium_lora/generated_images"

export REAL_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMG_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export EXTRA_INFO="SDV3-5"
export EXPERIMENT_TYPE="regular" # "regular" or "conditional" | "regular" calculates the metrics over all pathologies at once

export RESULTS_SAVEDIR="Results/"
export SHARDS_DIR="Results/saved_shards"
mkdir -p  $SHARDS_DIR

export NUM_SHARDS=4
export SHARD=0

# export BATCH_SIZE=128
export BATCH_SIZE=32
export NUM_WORKERS=4
export CUDA_VISIBLE_DEVICES=5

export REAL_CAPTION_COL="annotated_prompt"    # Using this instead of 'annotated_prompt'
# export SYNTHETIC_CAPTION_COL="prompt"  # Always remains this

echo "Calculating regular metrics for $EXTRA_INFO"

for (( shard=0; shard<NUM_SHARDS; shard++ )); do
    export SHARD=$shard
    echo "Calculating FID, KID, IS for shard $SHARD ..."
    ./scripts/fid_densenet.sh
done

# Combine the shards here
python tools/combine_shards_densenet.py --shards_dir=$SHARDS_DIR --extra_info=$EXTRA_INFO --output_dir=$RESULTS_SAVEDIR --delete_after_combining 