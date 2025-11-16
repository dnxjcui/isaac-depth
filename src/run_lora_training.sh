#!/bin/bash
# Example LoRA training script for Depth-Aware Isaac Model

# Set paths
MODEL_PATH="./isaac_model"
DATA_PATH="./data/train.json"
IMAGE_FOLDER="./data/images"
OUTPUT_DIR="./checkpoints/isaac-lora"
DEPTH_CHECKPOINT="./depth_anything_v2_vitl.pth"  # Optional

# Run training
python src/train_lora.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --use_depth True \
    --depth_checkpoint_path $DEPTH_CHECKPOINT \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --fp16 True \
    --gradient_checkpointing True \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --remove_unused_columns False

