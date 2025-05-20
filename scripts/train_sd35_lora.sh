RESOLUTION=1024
BATCH_SIZE=20
GRAD_ACC_STEPS=1
LR=1e-4
NUM_TRAIN_EPOCHS=20
VALIDATION_EPOCHS=5
DATASET="mimic"
TRAIN_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TRAIN.csv"
TEST_CSV="/pvc/Benchmarking-Synthetic-Data/MIMIC_Splits/LLAVARAD_ANNOTATIONS_TEST.csv"
IMG_DIR="/pvc/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
IMG_COL="path"
CAPTION_COL="annotated_prompt"

MODEL_NAME="stabilityai/stable-diffusion-3.5-medium"
OUTPUT_DIR="/pvc/Benchmarking-Synthetic-Data/OUTPUT_MIMIC_SD_3_5_lora"

accelerate launch --multi_gpu --main_process_port 35264 /pvc/Benchmarking-Synthetic-Data/tools/train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision="bf16" \
  --train_csv=$TRAIN_CSV \
  --test_csv=$TEST_CSV \
  --dataset_name=$DATASET \
  --image_column=$IMG_COL \
  --caption_column=$CAPTION_COL \
  --resolution=$RESOLUTION \
  --center_crop --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GRAD_ACC_STEPS \
  --gradient_checkpointing \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --validation_epochs=$VALIDATION_EPOCHS \
  --learning_rate=$LR \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --output_dir=$OUTPUT_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpoints_total_limit=2 --checkpointing_steps=1000 \