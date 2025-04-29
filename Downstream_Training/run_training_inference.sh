#!/bin/bash

## 1. Training on 15K REAL samples (baseline)
export TRAIN_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/training_data_15K_real.csv"
export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES/RadEdit"
export EXTRA_INFO="original_data"

## 2. Training on 15K Synthetic samples
# export TRAIN_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/training_data_15K_synthetic.csv"
# export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
# export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES/RadEdit"
# export EXTRA_INFO="syn_data_RadEdit"

## 3. Training on 15K Real + Synthetic samples
# export TRAIN_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/training_data_15K_mixed.csv"
# export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
# export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES/RadEdit"
# export EXTRA_INFO="syn_data_RadEdit"

## 4. Training on 30K Real + Synthetic samples
# export TRAIN_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/training_data_30K_augmented.csv"
# export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
# export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES/RadEdit"
# export EXTRA_INFO="syn_data_RadEdit"

export TEST_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"

export MODEL="resnet50" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
export BATCH_SIZE=758
export EPOCHS=10


export CHECKPOINT="Downstream_Training/checkpoints/${MODEL}_${EXTRA_INFO}.pth"

echo "Training the model..."
./Downstream_Training/run_training.sh > logs_ds_training.txt 2>&1
echo "Training Finished!!"

echo "Running inference..."
./Downstream_Training/run_inference.sh > logs_ds_inference.txt 2>&1
echo "Inference Finished!!"


