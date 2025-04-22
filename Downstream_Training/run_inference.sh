#!/bin/bash

python Downstream_Training/downstream_inference.py \
        --checkpoint=$CHECKPOINT \
        --test_csv=$TEST_CSV \
        --image_dir=$IMAGE_DIR \
        --save_predictions \
        --extra_info=$EXTRA_INFO