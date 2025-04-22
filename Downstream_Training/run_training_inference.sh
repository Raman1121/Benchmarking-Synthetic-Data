#!/bin/bash

export TRAIN_CSV="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/LLAVARAD_ANNOTATIONS_TRAIN.csv"
export TEST_CSV="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/ANNOTATED_CSV_FILES/LLAVARAD_ANNOTATIONS_TEST.csv"
export IMAGE_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"

export MODEL="vit_base_patch16_224.orig_in21k_ft_in1k" #resnet50, vit_base_patch16_224.orig_in21k_ft_in1k
export BATCH_SIZE=1024
export EPOCHS=20

export EXTRA_INFO="original_data"
export CHECKPOINT="Downstream_Training/checkpoints/{$MODEL}_best_model.pth"

echo "Training the model..."
./Downstream_Training/run_training.sh > logs_downstream_classification.txt 2>&1
echo "Training Finished!!"

echo "Running inference..."
./Downstream_Training/run_inference.sh > logs_downstream_classification_inference.txt 2>&1
echo "Inference Finished!!"


