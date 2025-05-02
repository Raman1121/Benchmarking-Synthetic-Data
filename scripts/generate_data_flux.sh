#!/bin/bash

TEST_PROMPTS_PATH="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_test_prompts.txt"
# TEST_PROMPTS_PATH="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_train_prompts_20K.txt"

CKPT_NAME="flux_lora.safetensors"
MODEL_PATH="/pvc/ai-toolkit/output/flux_lora"
SAVE_DIR=${MODEL_PATH}/generated_images
mkdir -p $SAVE_DIR

BATCH_SIZE=48
 
accelerate launch tools/generate_data_flux.py \
                        --ckpt_dir $MODEL_PATH \
                        --ckpt_name $CKPT_NAME \
                        --savedir $SAVE_DIR \
                        --batch_size $BATCH_SIZE \
                        --test_prompts_path $TEST_PROMPTS_PATH
