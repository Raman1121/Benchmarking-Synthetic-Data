#!/bin/bash

SAVE_DIR="/pvc/SYNTHETIC_IMAGES/RadEdit/"
# EXTRA_INFO=""

TEST_PROMPTS_PATH="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/mimic_train_prompts_20K.txt"

BATCH_SIZE=64

python tools/generate_radedit.py \
        --test_prompts_path $TEST_PROMPTS_PATH \
        --savedir $SAVE_DIR \
        --batch_size $BATCH_SIZE \