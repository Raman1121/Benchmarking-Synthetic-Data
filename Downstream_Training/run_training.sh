#!/bin/bash

source /raid/s2198939/miniconda3/bin/activate sana

export TRAIN_CSV="MIMIC_Splits/LLAVARAD_ANNOTATIONS_TRAIN.csv"
export IMAGE_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export MODEL="vit_base_patch16_224.orig_in21k_ft_in1k" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
export BATCH_SIZE=64
export EPOCHS=15

export CUDA_VISIBLE_DEVICES=0,1,2,4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python Downstream_Training/train_downstream_classification.py \
                                            --batch_size=$BATCH_SIZE \
                                            --epochs=$EPOCHS \
                                            --model_name=$MODEL \
                                            --csv_path=$TRAIN_CSV \
                                            --image_dir=$IMAGE_DIR \
                                            # --debug
