#!/bin/bash

MODEL_PATH="/pvc/Benchmarking-Synthetic-Data/OUTPUT_MIMIC_SD_V1_5/IID/512/SD-V1-5_IID_512"
SAVE_DIR="/pvc/SYNTHETIC_IMAGES/"
# EXTRA_INFO=""

TEST_PROMPTS_PATH="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_train_prompts_20K.txt"

BATCH_SIZE=64

accelerate launch tools/generate_sd.py --model_path $MODEL_PATH \
                                        --savedir $SAVE_DIR \
                                        --batch_size $BATCH_SIZE \
                                        --test_prompts_path $TEST_PROMPTS_PATH
