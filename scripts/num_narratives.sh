#!/bin/bash
# Run number-of-narratives generation in batch
# Usage: bash scripts/num_narratives.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root so relative paths work
cd "$PROJECT_ROOT"

# Define an array of model names.
model_list=(
    "unsloth_qwen_0.5B"
    "unsloth_qwen3_0.6B"
    "unsloth_llama_1B-Instruct"
    "unsloth_deepseek_r1_qwen_1.5B"
    "unsloth_qwen3_1.7B"
    "unsloth_llama_3B-Instruct"
    "unsloth_qwen_3B"
    #"unsloth_qwen3_4B"
)

DATASETS=(
    "adult"
    "titanic"
    "california"
    "diabetes"
)

TEMPERATURES=(
    "0.1"
    "0.6"
    "1.0"
)

# Iterate over each model and run num_narratives.py accordingly.
for dataset in "${DATASETS[@]}"; do
    for model in "${model_list[@]}"; do
        for temperature in "${TEMPERATURES[@]}"; do
            echo "Running num_narratives.py with model '$model', dataset '$dataset', temperature '$temperature'..."
            python num_narratives.py --dataset="$dataset" --worker_model_name="$model" --temperature="$temperature"
        done
    done
done

