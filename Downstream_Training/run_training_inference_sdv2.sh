#!/bin/bash

########################################### 1. Training on 20K REAL samples (baseline) ###########################################
# echo "RUNNING EXPERIMENT: ALL REAL DATA"
# export MODEL="resnet50" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
# export T2I_MODEL="SD-V2"
# export BATCH_SIZE=758
# export TRAIN_CSV="/pvc/SYNTHETIC_IMAGES_NEW/SD-V2/generations_with_metadata.csv"
# export TEST_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
# export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
# export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES_NEW/SD-V2/generations_with_metadata.csv"
# export TRAINING_SETTING="all_original"
# export EXTRA_INFO=$TRAINING_SETTING
# export EPOCHS=20

# export CHECKPOINT="Downstream_Training/checkpoints/${MODEL}_${EXTRA_INFO}.pth"

# echo "Training the model..."
# chmod +x Downstream_Training/run_training.sh
# ./Downstream_Training/run_training.sh > logs_ds_training_original.txt 2>&1
# echo "Training Finished!!"

# echo "Running inference..."
# chmod +x Downstream_Training/run_inference.sh
# ./Downstream_Training/run_inference.sh > logs_ds_inference_original.txt 2>&1
# echo "Inference Finished!!"

########################################### 2. Training on 20K Synthetic samples ###########################################
echo "RUNNING EXPERIMENT: ALL SYNTHETIC DATA"
export MODEL="resnet50" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
export T2I_MODEL="SD-V2"
export BATCH_SIZE=758
export TRAIN_CSV="/pvc/SYNTHETIC_IMAGES_NEW/SD-V2/generations_with_metadata.csv"
export TEST_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES_NEW/SD-V2"
export TRAINING_SETTING="all_synthetic"
export export EXTRA_INFO=$TRAINING_SETTING

export EPOCHS=20

export CHECKPOINT="Downstream_Training/checkpoints/${MODEL}_${EXTRA_INFO}_${T2I_MODEL}.pth"

echo "Training the model..."
chmod +x Downstream_Training/run_training.sh
./Downstream_Training/run_training.sh > logs_ds_training_synthetic_${T2I_MODEL}.txt 2>&1
echo "Training Finished!!"

echo "Running inference..."
chmod +x Downstream_Training/run_inference.sh
./Downstream_Training/run_inference.sh > logs_ds_inference_synthetic_${T2I_MODEL}.txt 2>&1
echo "Inference Finished!!"

########################################### 3. Training on 20K Real + Synthetic samples ###########################################
# echo "RUNNING EXPERIMENT: MIXED DATA"
# export MODEL="resnet50" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
# export T2I_MODEL="SD-V2"
# export BATCH_SIZE=758
# export TRAIN_CSV="/pvc/SYNTHETIC_IMAGES_NEW/SD-V2/generations_with_metadata.csv"
# export TEST_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
# export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
# export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES_NEW/SD-V2"
# export TRAINING_SETTING="mixed"
# export EXTRA_INFO=$TRAINING_SETTING

# export EPOCHS=20

# export CHECKPOINT="Downstream_Training/checkpoints/${MODEL}_${EXTRA_INFO}_${T2I_MODEL}.pth"

# echo "Training the model..."
# chmod +x Downstream_Training/run_training.sh
# ./Downstream_Training/run_training.sh > logs_ds_training_mixed_${T2I_MODEL}.txt 2>&1
# echo "Training Finished!!"

# echo "Running inference..."
# chmod +x Downstream_Training/run_inference.sh
# ./Downstream_Training/run_inference.sh > logs_ds_inference_mixed_${T2I_MODEL}.txt 2>&1
# echo "Inference Finished!!"

########################################### 4. Training on 40K Real + Synthetic samples ###########################################
# echo "RUNNING EXPERIMENT: AUGMENTED DATA"
# export MODEL="resnet50" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
# export T2I_MODEL="SD-V2"
# export BATCH_SIZE=758
# export TRAIN_CSV="/pvc/SYNTHETIC_IMAGES_NEW/SD-V2/generations_with_metadata.csv"
# export TEST_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
# export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
# export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES_NEW/SD-V2"
# export TRAINING_SETTING="augmented"
# export EXTRA_INFO=$TRAINING_SETTING

# export EPOCHS=20


# export CHECKPOINT="Downstream_Training/checkpoints/${MODEL}_${EXTRA_INFO}_${T2I_MODEL}.pth"

# echo "Training the model..."
# chmod +x Downstream_Training/run_training.sh
# ./Downstream_Training/run_training.sh > logs_ds_training_augmented_${T2I_MODEL}.txt 2>&1
# echo "Training Finished!!"

# echo "Running inference..."
# chmod +x Downstream_Training/run_inference.sh
# ./Downstream_Training/run_inference.sh > logs_ds_inference_augmented_${T2I_MODEL}.txt 2>&1
# echo "Inference Finished!!"