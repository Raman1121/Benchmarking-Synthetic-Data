#!/bin/bash

python Downstream_Training/train_downstream_classification.py \
        --batch_size=$BATCH_SIZE \
        --epochs=$EPOCHS \
        --model_name=$MODEL \
        --csv_path=$TRAIN_CSV \
        --real_image_dir=$REAL_IMAGE_DIR \
        --synthetic_image_dir=$SYNTHETIC_IMG_DIR \
        --extra_info=$EXTRA_INFO \