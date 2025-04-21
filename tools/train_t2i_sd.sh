#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

RESOLUTION=512
BATCH_SIZE=224
GRAD_ACC_STEPS=1
LR=5e-6
WARMUP_STEPS=500
NUM_TRAIN_EPOCHS=20
VALIDATION_EPOCHS=5
# MAX_TRAIN_STEPS=1000
TRAINING_SETTING="IID"
DATASET="mimic"
TRAIN_CSV="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/LLAVARAD_ANNOTATIONS_TRAIN.csv"
TEST_CSV="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/LLAVARAD_ANNOTATIONS_TEST.csv"
IMG_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
IMG_COL="path"
CAPTION_COL="annotated_prompt"

MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="OUTPUT_MIMIC_SD_V1_4"

accelerate launch --main_process_port 12345 tools/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision="fp16" \
  --train_csv=$TRAIN_CSV \
  --test_csv=$TEST_CSV \
  --dataset_name=$DATASET \
  --train_data_dir=$IMG_DIR \
  --image_column=$IMG_COL \
  --caption_column=$CAPTION_COL \
  --resolution=$RESOLUTION \
  --center_crop --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GRAD_ACC_STEPS \
  --gradient_checkpointing \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --validation_epochs=$VALIDATION_EPOCHS \
  --learning_rate=$LR \
  --training_setting=$TRAINING_SETTING \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=$WARMUP_STEPS \
  --output_dir=$OUTPUT_DIR \
  --checkpoints_total_limit=1 --checkpointing_steps=1000 \
#   --enable_xformers_memory_efficient_attention \