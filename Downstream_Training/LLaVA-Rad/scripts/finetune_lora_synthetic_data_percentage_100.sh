#!/bin/bash

# Set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1

model_base=lmsys/vicuna-7b-v1.5
# output_dir="${1:-./checkpoints_test}"

PROJECTOR="/pvc/mm_projector.bin" # generated using pretrain.sh
vision_tower="biomedclip_cxr_518"
vision_tower_config="llava/model/multimodal_encoder/open_clip_encoder/model_configs/biomedclip_cxr_518.json"
vision_tower_checkpoint="biomedclipcxr_518_checkpoint.pt"
################## VICUNA ##################


################## Synthetic Data Details ##################
ALL_T2I_MODELS=("SD-V1-4" "SD-V1-5" "SD-V2" "SD-V2-1" "SD-V3-5" "pixart_sigma" "radedit" "sana" "lumina" "flux")

for t2i_model in "${ALL_T2I_MODELS[@]}"; do
    export T2I_MODEL=$t2i_model
    export DATA_PERCENTAGE=100
    export output_dir="checkpoints_${T2I_MODEL}"

    data_path=/pvc/SYNTHETIC_IMAGES_NEW/$T2I_MODEL/generations_with_metadata.csv
    loader="default"
    image_folder=/pvc/SYNTHETIC_IMAGES_NEW/$T2I_MODEL

    ################## Run name ##################
    epoch="${2:-5}"
    bsz="${3:-16}"

    lr="1e-4"
    schedule="lora-${epoch}e"
    export run_name="${vision_tower}-${schedule}-${lr}-$(date +%Y%m%d%H%M%S)"

    echo $run_name > run_name
    echo "Epoch: $epoch"
    echo "Batch size: $bsz"
    echo "Learning rate: $lr"
    # echo "Num Samples: $num_samples"
    echo "T2I Model: $T2I_MODEL"
    echo "Data Percentage: $DATA_PERCENTAGE"
    ################## Run name ##################


    # Batch size is set for 4-GPU machines.
    python llava/train/train_mem.py \
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
        --run_name ${run_name} \
        --t2i_model ${T2I_MODEL} \
        --data_percentage ${DATA_PERCENTAGE}


    ################################# INFERENCE #################################

    echo "Running Inference!!!"

    model_base=lmsys/vicuna-7b-v1.5
    model_path="${output_dir}/llavarad_lora_${T2I_MODEL}_percentage_${DATA_PERCENTAGE}"

    prediction_dir="${model_path}/results/llavarad"
    prediction_file=$prediction_dir/test

    run_name="llavarad"
    query_file=/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/LLavA-Rad-Annotations/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json
    image_folder=/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files

    loader="mimic_test_findings"
    conv_mode="v1"

    # CHUNKS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    CHUNK_IDX=0
    NUM_CHUNKS=1

    python -m llava.eval.model_mimic_cxr \
            --query_file ${query_file} \
            --loader ${loader} \
            --image_folder ${image_folder} \
            --conv_mode ${conv_mode} \
            --prediction_file ${prediction_file}_${idx}.jsonl \
            --temperature 0 \
            --model_path ${model_path} \
            --model_base ${model_base} \
            --chunk_idx ${CHUNK_IDX} \
            --num_chunks ${NUM_CHUNKS} \
            --batch_size 32 \
            --group_by_length &


    wait
    echo "All done!"

    cat ${prediction_file}_*.jsonl > "$prediction_dir/mimic_cxr_preds_${T2I_MODEL}_percentage_${DATA_PERCENTAGE}.jsonl"
done