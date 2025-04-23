#!/bin/bash

source /raid/s2198939/miniconda3/bin/activate sana

MODEL_PATH="/raid/s2198939/medical-diffusion-classifier/OUTPUT_MIMIC_SD_V1_4/IID/512/SD-V1-4_IID_512"
SAVE_DIR=${MODEL_PATH}/generated_images
mkdir -p $SAVE_DIR

TEST_PROMPTS_PATH="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_test_prompts.txt"

BATCH_SIZE=8
CUDA_VISIBLE_DEVICES=7 

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch tools/generate_sd.py --model_path $MODEL_PATH \
                                                                 --savedir $SAVE_DIR \
                                                                 --batch_size $BATCH_SIZE \
                                                                 --test_prompts_path $TEST_PROMPTS_PATH
