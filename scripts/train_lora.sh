#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-Omni-7B"
GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/train.py \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/data.json \
    --media_folder /path/to/media/folder \
    --output_dir output/omni_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate 1e-4 \
    --vision_lr 2e-6 \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --lr_scheduler_type "cosine" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing True \
    --deepspeed scripts/deepspeed_configs/zero3.json \
    --report_to tensorboard \
    --dataloader_num_workers 4