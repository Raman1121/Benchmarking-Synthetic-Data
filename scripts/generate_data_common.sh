#!/bin/bash

#################### Sana ####################
# export MODEL_PATH="/pvc/Benchmarking-Synthetic-Data/Sana_Diffusers"
# export MODEL_NAME="sana"
# export EXTRA_INFO="sana"

# export REAL_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv"
# export SAVE_DIR="/pvc/SYNTHETIC_IMAGES_NEW/"

# export BATCH_SIZE=128

# python tools/generate_data_common.py \
#     --model_path=$MODEL_PATH \
#     --model_name=$MODEL_NAME \
#     --extra_info=$EXTRA_INFO \
#     --real_csv=$REAL_CSV \
#     --savedir=$SAVE_DIR \
#     --batch_size=$BATCH_SIZE \

#################### RadEdit ####################
# export MODEL_PATH="/pvc/Sana/output/Sana_Diffusers/"
# export MODEL_NAME="radedit"
# export EXTRA_INFO="radedit"

# export REAL_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv"
# export SAVE_DIR="/pvc/SYNTHETIC_IMAGES_NEW/"

# export BATCH_SIZE=128

# python tools/generate_data_common.py \
#     --model_path=$MODEL_PATH \
#     --model_name=$MODEL_NAME \
#     --extra_info=$EXTRA_INFO \
#     --real_csv=$REAL_CSV \
#     --savedir=$SAVE_DIR \
#     --batch_size=$BATCH_SIZE \

#################### SD-V1-4 ####################
# export MODEL_PATH="/pvc/Benchmarking-Synthetic-Data/OUTPUT_MIMIC_SD_V1_4/IID/512/SD-V1-4_IID_512"
# export MODEL_NAME="SD-V1-4"
# export EXTRA_INFO="SD-V1-4"

# export REAL_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv"
# export SAVE_DIR="/pvc/SYNTHETIC_IMAGES_NEW/"

# export BATCH_SIZE=64

# python tools/generate_data_common.py \
#     --model_path=$MODEL_PATH \
#     --model_name=$MODEL_NAME \
#     --extra_info=$EXTRA_INFO \
#     --real_csv=$REAL_CSV \
#     --savedir=$SAVE_DIR \
#     --batch_size=$BATCH_SIZE \

#################### SD-V1-5 ####################
export MODEL_PATH="/pvc/Benchmarking-Synthetic-Data/OUTPUT_MIMIC_SD_V1_5/IID/512/SD-V1-5_IID_512"
export MODEL_NAME="SD-V1-5"
export EXTRA_INFO="SD-V1-5"

export REAL_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv"
export SAVE_DIR="/pvc/SYNTHETIC_IMAGES_NEW/"

export BATCH_SIZE=64

python tools/generate_data_common.py \
    --model_path=$MODEL_PATH \
    --model_name=$MODEL_NAME \
    --extra_info=$EXTRA_INFO \
    --real_csv=$REAL_CSV \
    --savedir=$SAVE_DIR \
    --batch_size=$BATCH_SIZE \

#################### SD-V2 ####################
# export MODEL_PATH="/pvc/Benchmarking-Synthetic-Data/OUTPUT_MIMIC_SD_V2/IID/512/SD-V2_IID_512"
# export MODEL_NAME="SD-V2"
# export EXTRA_INFO="SD-V2"

# export REAL_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv"
# export SAVE_DIR="/pvc/SYNTHETIC_IMAGES_NEW/"

# export BATCH_SIZE=64

# python tools/generate_data_common.py \
#     --model_path=$MODEL_PATH \
#     --model_name=$MODEL_NAME \
#     --extra_info=$EXTRA_INFO \
#     --real_csv=$REAL_CSV \
#     --savedir=$SAVE_DIR \
#     --batch_size=$BATCH_SIZE \

#################### SD-V2-1 ####################
# export MODEL_PATH="/pvc/Benchmarking-Synthetic-Data/OUTPUT_MIMIC_SD_V2_1/IID/512/SD-V2-1_IID_512"
# export MODEL_NAME="SD-V2-1"
# export EXTRA_INFO="SD-V2-1"

# export REAL_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv"
# export SAVE_DIR="/pvc/SYNTHETIC_IMAGES_NEW/"

# export BATCH_SIZE=64

# python tools/generate_data_common.py \
#     --model_path=$MODEL_PATH \
#     --model_name=$MODEL_NAME \
#     --extra_info=$EXTRA_INFO \
#     --real_csv=$REAL_CSV \
#     --savedir=$SAVE_DIR \
#     --batch_size=$BATCH_SIZE \

#################### Pixart Sigma ####################
# export MODEL_PATH="/pvc/Benchmarking-Synthetic-Data/Pixart_diffusers"
# export MODEL_NAME="pixart_sigma"
# export EXTRA_INFO="pixart_sigma"

# export REAL_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv"
# export SAVE_DIR="/pvc/SYNTHETIC_IMAGES_NEW/"

# export BATCH_SIZE=128

# python tools/generate_data_common.py \
#     --model_path=$MODEL_PATH \
#     --model_name=$MODEL_NAME \
#     --extra_info=$EXTRA_INFO \
#     --real_csv=$REAL_CSV \
#     --savedir=$SAVE_DIR \
#     --batch_size=$BATCH_SIZE \

#################### SDV-3.5 LoRA ####################
# export MODEL_PATH="/pvc/ai-toolkit/output/sd3-5_medium_lora"
# export MODEL_NAME="SD-V3-5"
# export EXTRA_INFO="SD-V3-5"

# export REAL_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv"
# export SAVE_DIR="/pvc/SYNTHETIC_IMAGES_NEW/"

# export BATCH_SIZE=96

# python tools/generate_data_common.py \
#     --model_path=$MODEL_PATH \
#     --model_name=$MODEL_NAME \
#     --extra_info=$EXTRA_INFO \
#     --real_csv=$REAL_CSV \
#     --savedir=$SAVE_DIR \
#     --batch_size=$BATCH_SIZE \

#################### Lumina ####################
# export MODEL_PATH="/pvc/ai-toolkit/output/lumina2_lora"
# export MODEL_NAME="lumina"
# export EXTRA_INFO="lumina"

# export REAL_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/Downstream_Classification_Files/training_data_20K.csv"
# export SAVE_DIR="/pvc/SYNTHETIC_IMAGES_NEW/"

# export BATCH_SIZE=128

# python tools/generate_data_common.py \
#     --model_path=$MODEL_PATH \
#     --model_name=$MODEL_NAME \
#     --extra_info=$EXTRA_INFO \
#     --real_csv=$REAL_CSV \
#     --savedir=$SAVE_DIR \
#     --batch_size=$BATCH_SIZE \

#################### FLUX ####################
# TBD!!!!