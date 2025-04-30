#!/bin/bash

source /raid/s2198939/miniconda3/bin/activate sana

MODEL_PATH="/raid/s2198939/medical-diffusion-classifier/OUTPUT_MIMIC_LUMINA"
SAVE_DIR=${MODEL_PATH}/generated_images
mkdir -p $SAVE_DIR

TEST_PROMPTS_PATH="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_test_prompts.txt"

CKPT_NAME="lumina2_lora.safetensors"

BATCH_SIZE=1
CUDA_VISIBLE_DEVICES=2 

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch tools/generate_data_lumina.py \
                                                                --ckpt_dir $MODEL_PATH \
                                                                --ckpt_name $CKPT_NAME \
                                                                --savedir $SAVE_DIR \
                                                                --batch_size $BATCH_SIZE \
                                                                --test_prompts_path $TEST_PROMPTS_PATH
