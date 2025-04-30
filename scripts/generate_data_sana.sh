#!/bin/bash

MODEL_PATH="/data/Sana/output Sana_Diffusers/"
SAVE_DIR="/pvc/SYNTHETIC_IMAGES/"
EXTRA_INFO="Sana"

TEST_PROMPTS_PATH="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_train_prompts_20K.txt"

BATCH_SIZE=64

accelerate launch tools/generate_sd.py --model_path $MODEL_PATH \
                                        --savedir $SAVE_DIR \
                                        --batch_size $BATCH_SIZE \
                                        --test_prompts_path $TEST_PROMPTS_PATH \
                                        --extra_info $EXTRA_INFO \
