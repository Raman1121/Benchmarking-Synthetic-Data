#!/bin/bash

set -e
set -o pipefail

METADATA_CSV="/pvc/SynthCheX/sana_epoch50/generations_with_metadata.csv"
IMAGE_DIR="/pvc/SynthCheX/sana_epoch50/"
OUTPUT_DIR="/pvc/SynthChex/"

IMAGE_COL="synthetic_filename"
CAPTION_COL="annotated_prompt"
LABELS_COL="chexpert_labels"

NUM_SHARDS=10000
SHARD=0

python scripts/assess_data_quality.py \
    --metadata_csv $METADATA_CSV \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --num_shards $NUM_SHARDS \
    --shard $SHARD \
    --image_col $IMAGE_COL \
    --caption_col $CAPTION_COL \
    --labels_col $LABELS_COL \