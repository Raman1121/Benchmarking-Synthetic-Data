#!/bin/bash

TEST_PROMPTS_PATH="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_test_prompts.txt"

CKPT_NAME="lumina2_lora.safetensors"
MODEL_PATH="/pvc/ai-toolkit/output/lumina2_lora"
SAVE_DIR=${MODEL_PATH}/generated_images
mkdir -p $SAVE_DIR

BATCH_SIZE=64
 
accelerate launch tools/generate_data_lumina.py \
                        --ckpt_dir $MODEL_PATH \
                        --ckpt_name $CKPT_NAME \
                        --savedir $SAVE_DIR \
                        --batch_size $BATCH_SIZE \
                        --test_prompts_path $TEST_PROMPTS_PATH
