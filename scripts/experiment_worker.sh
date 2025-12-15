#!/bin/bash
# Run worker model experiments in batch
# Usage: bash scripts/experiment_worker.sh

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
)

# Iterate over each model configuration and run experiment.py accordingly.
for model in "${model_list[@]}"; do
    echo "Running experiment.py with model '$model'..."
    python experiment.py --dataset=adult --worker_model_name="$model" --analyze_feasibility
    python experiment.py --dataset=adult --worker_model_name="$model" --analyze_feasibility --worker_fine_tuned
done

