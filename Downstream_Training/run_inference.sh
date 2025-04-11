#!/bin/bash

source /raid/s2198939/miniconda3/bin/activate sana

export CHECKPOINT="Downstream_Training/checkpoints/resnet50_best_model.pth"
export TEST_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export IMAGE_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export CUDA_VISIBLE_DEVICES=2

export EXTRA_INFO="original_data"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python Downstream_Training/downstream_inference.py \
                                                --checkpoint=$CHECKPOINT \
                                                --test_csv=$TEST_CSV \
                                                --image_dir=$IMAGE_DIR \
                                                --save_predictions \
                                                --extra_info=$EXTRA_INFO