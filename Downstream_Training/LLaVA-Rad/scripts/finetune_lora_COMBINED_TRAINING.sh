#!/bin/bash

# Set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1

model_base=lmsys/vicuna-7b-v1.5
# output_dir="${1:-./checkpoints_test}"

# PROJECTOR="/pvc/mm_projector.bin" # generated using pretrain.sh
PROJECTOR="/pvc/Benchmarking-Synthetic-Data/Downstream_Training/LLaVA-Rad/CHECKPOINTS_NEW_AUGMENTED_sana/llavarad_lora_sana_percentage_100/mm_projector_cleaned.bin" # Using a new projector TFS on mixed data
vision_tower="biomedclip_cxr_518"
vision_tower_config="llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json"
vision_tower_checkpoint="biomedclipcxr_518_checkpoint.pt"
################## VICUNA ##################

################## Data ##################
t2i_model="sana"
export T2I_MODEL=$t2i_model
export DATA_PERCENTAGE=100
export output_dir="checkpoints_COMBINED_${T2I_MODEL}"

# data_path=/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/REAL_SYN_COMBINED_CSV_RRG/combined_CSV_${T2I_MODEL}.json
data_path="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/REAL_SYN_COMBINED_CSV_RRG_SUBSET_50K/combined_CSV_${T2I_MODEL}_SUBSET_50K.json"
loader="default"
image_folder=""   # Keeping this None since we have already appended the root path in the CSV/JSON file
################## Data ##################

################## Run name ##################
epoch="${2:-10}"
bsz="${3:-16}"
lr="1e-4"
schedule="COMBINED-lora-${epoch}e"
export run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"
echo $run_name > run_name
################## Run name ##################

# Batch size is set for 4-GPU machines.
CUDA_VISIBLE_DEVICES=0
    deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_alpha 128 \
    --model_name_or_path ${model_base} \
    --version $PROMPT_VERSION \
    --data_path ${data_path} \
    --loader ${loader} \
    --image_folder "${image_folder}" \
    --vision_tower ${vision_tower} \
    --vision_tower_config ${vision_tower_config} \
    --vision_tower_checkpoint ${vision_tower_checkpoint} \
    --pretrain_mm_mlp_adapter ${PROJECTOR} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${output_dir}/${run_name} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${bsz} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --run_name ${run_name} \
    --t2i_model ${T2I_MODEL} \
    --data_percentage ${DATA_PERCENTAGE}