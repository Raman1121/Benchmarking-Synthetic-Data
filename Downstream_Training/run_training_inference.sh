#!/bin/bash

## 1. Training on 20K REAL samples (baseline)
export TRAIN_CSV="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/training_data_15K_real.csv"
export IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0" # Directory where the real images are located => original MIMIC dataset
export EXTRA_INFO="original_data"

## 2. Training on 20K Synthetic samples
# export TRAIN_CSV="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/"
# export IMAGE_DIR="/pvc/SYNTHETIC_IMAGES/Pixart_Sigma" # Directory where the synthetic images are located
# export EXTRA_INFO="syn_data_Pixart_Sigma"

export TEST_CSV="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/LLAVARAD_ANNOTATIONS_TEST.csv"

export MODEL="resnet50" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
export BATCH_SIZE=512
export EPOCHS=10


export CHECKPOINT="Downstream_Training/checkpoints/{$MODEL}_{$EXTRA_INFO}.pth"

echo "Training the model..."
./Downstream_Training/run_training.sh > logs_ds_training.txt 2>&1
echo "Training Finished!!"

echo "Running inference..."
./Downstream_Training/run_inference.sh > logs_ds_inference.txt 2>&1
echo "Inference Finished!!"


