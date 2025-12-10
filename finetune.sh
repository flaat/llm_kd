#!/bin/bash
# Define an array of model names.
model_list=(
    "unsloth_qwen_0.5B"
    "unsloth_qwen3_0.6B"
    "unsloth_llama_1B-Instruct"
    "unsloth_deepseek_r1_qwen_1.5B"
    "unsloth_qwen3_1.7B"
    "unsloth_llama_3B-Instruct"
    "unsloth_qwen_3B"
    "unsloth_qwen3_4B-Instruct"
)

# Iterate over each model configuration and run tests.py accordingly.
for model in "${model_list[@]}"; do
    echo "⚙️ Running tests.py with model '$model'..."
    python finetune.py --model_name="$model" --dataset="adult" 
done