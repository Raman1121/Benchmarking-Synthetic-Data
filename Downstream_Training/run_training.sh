#!/bin/bash

python Downstream_Training/train_downstream_classification.py \
        --batch_size=$BATCH_SIZE \
        --epochs=$EPOCHS \
        --model_name=$MODEL \
        --csv_path=$TRAIN_CSV \
        --image_dir=$IMAGE_DIR \
        --debug
