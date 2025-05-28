#!/bin/bash

MODEL_BASE="Qwen/Qwen2.5-Omni-7B"
LORA_PATH="output/omni_lora/checkpoint-1000"
SAVE_PATH="output/omni_merged"

python src/merge_lora.py \
    --model-path $LORA_PATH \
    --model-base $MODEL_BASE \
    --save-path $SAVE_PATH