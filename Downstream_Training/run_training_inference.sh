#!/bin/bash

########################################### 1. Training on 15K REAL samples (baseline) ###########################################
# export MODEL="resnet50" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
# export T2I_MODEL="Sana"
# export BATCH_SIZE=758
# export TRAIN_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/training_data_15K_real.csv"
# export TEST_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
# export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
# export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES/Sana"
# export EXTRA_INFO="original_data"
# export EPOCHS=10

# export CHECKPOINT="Downstream_Training/checkpoints/${MODEL}_${EXTRA_INFO}.pth"

# echo "Training the model..."
# chmod +x Downstream_Training/run_training.sh
# ./Downstream_Training/run_training.sh > logs_ds_training_original.txt 2>&1
# echo "Training Finished!!"

# echo "Running inference..."
# chmod +x Downstream_Training/run_inference.sh
# ./Downstream_Training/run_inference.sh > logs_ds_inference_original.txt 2>&1
# echo "Inference Finished!!"

########################################### 2. Training on 15K Synthetic samples ###########################################
export MODEL="resnet50" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
export T2I_MODEL="Sana"
export BATCH_SIZE=758
export TRAIN_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Eval/${T2I_MODEL}/training_data_15K_synthetic.csv"
export TEST_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES/Sana"
export EXTRA_INFO="syn_data_${T2I_MODEL}"
export EPOCHS=10

export CHECKPOINT="Downstream_Training/checkpoints/${MODEL}_${EXTRA_INFO}.pth"

echo "Training the model..."
chmod +x Downstream_Training/run_training.sh
./Downstream_Training/run_training.sh > logs_ds_training_synthetic.txt 2>&1
echo "Training Finished!!"

echo "Running inference..."
chmod +x Downstream_Training/run_inference.sh
./Downstream_Training/run_inference.sh > logs_ds_inference_synthetic.txt 2>&1
echo "Inference Finished!!"

########################################### 3. Training on 15K Real + Synthetic samples ###########################################
# export MODEL="resnet50" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
# export T2I_MODEL="Sana"
# export BATCH_SIZE=758
# export TRAIN_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Eval/${T2I_MODEL}/training_data_15K_mixed.csv"
# export TEST_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
# export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
# export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES/Sana"
# export EXTRA_INFO="mixed_data_${T2I_MODEL}"
# export EPOCHS=10

# export CHECKPOINT="Downstream_Training/checkpoints/${MODEL}_${EXTRA_INFO}.pth"

# echo "Training the model..."
# chmod +x Downstream_Training/run_training.sh
# ./Downstream_Training/run_training.sh > logs_ds_training_mixed.txt 2>&1
# echo "Training Finished!!"

# echo "Running inference..."
# chmod +x Downstream_Training/run_inference.sh
# ./Downstream_Training/run_inference.sh > logs_ds_inference_mixed.txt 2>&1
# echo "Inference Finished!!"

########################################### 4. Training on 30K Real + Synthetic samples ###########################################
# export MODEL="resnet50" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
# export T2I_MODEL="Sana"
# export BATCH_SIZE=758
# export TRAIN_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Eval/${T2I_MODEL}/training_data_30K_augmented.csv"
# export TEST_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
# export REAL_IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
# export SYNTHETIC_IMG_DIR="/pvc/SYNTHETIC_IMAGES/Sana"
# export EXTRA_INFO="augmented_data_${T2I_MODEL}"
# export EPOCHS=20


# export CHECKPOINT="Downstream_Training/checkpoints/${MODEL}_${EXTRA_INFO}.pth"

# echo "Training the model..."
# chmod +x Downstream_Training/run_training.sh
# ./Downstream_Training/run_training.sh > logs_ds_training_augmented.txt 2>&1
# echo "Training Finished!!"

# echo "Running inference..."
# chmod +x Downstream_Training/run_inference.sh
# ./Downstream_Training/run_inference.sh > logs_ds_inference_augmented.txt 2>&1
# echo "Inference Finished!!"


