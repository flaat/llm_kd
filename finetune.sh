#!/bin/bash
# Define an array of model names.
model_list=(
    "unsloth_deepseek_r1_qwen_1.5B"
    "unsloth_qwen_0.5B"
    "unsloth_qwen_3B"
    "unsloth_deepseek_r1_qwen_7B"
)

# Iterate over each model configuration and run tests.py accordingly.
for model in "${model_list[@]}"; do
    echo "Running tests.py with model '$model'..."
    python tests.py --model_name="$model" --dataset="titanic"
done