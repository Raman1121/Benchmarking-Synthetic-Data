#!/bin/bash

# Set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1

model_base=lmsys/vicuna-7b-v1.5
output_dir="${1:-./checkpoints_synthetic_radedit}"

PROJECTOR="/pvc/mm_projector.bin" # generated using pretrain.sh
vision_tower="biomedclip_cxr_518"
vision_tower_config="llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json"
vision_tower_checkpoint="biomedclipcxr_518_checkpoint.pt"
################## VICUNA ##################


################## Synthetic Data ##################
# data_path=/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json
data_path=/pvc/SYNTHETIC_IMAGES_NEW/radedit/generations_with_metadata.csv
# loader="mimic_train_findings"
loader="default"
# image_folder=/pvc/Benchmarking-Synthetic-Data/assets/synthetic_images/
image_folder=/pvc/SYNTHETIC_IMAGES_NEW/radedit/

################## Synthetic Data ##################

################## Run name ##################
epoch="${2:-3}"
bsz="${3:-16}"
# epoch=5
# bsz=48


lr="1e-4"
schedule="lora-${epoch}e"
export run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"
echo $run_name > run_name
echo "Epoch: $epoch"
echo "Batch size: $bsz"
echo "Learning rate: $lr"
################## Run name ##################


# Batch size is set for 4-GPU machines.
    deepspeed llava/train/train_mem.py \
    --deepspeed /pvc/Benchmarking-Synthetic-Data/Downstream_Training/LLaVA-Rad/scripts/zero2.json \
    --lora_enable True \
    --lora_alpha 128 \
    --model_name_or_path ${model_base} \
    --version $PROMPT_VERSION \
    --data_path ${data_path} \
    --loader ${loader} \
    --image_folder ${image_folder} \
    --finetune_only_with_synthetic_data True \
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
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --run_name ${run_name}

    # NOTE: # Remove "--finetune_only_with_synthetic_data True" line to finetune on both real and synthetic data
