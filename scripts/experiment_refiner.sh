#!/bin/bash
# Run refiner model experiments in batch
# Usage: bash scripts/experiment_refiner.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root so relative paths work
cd "$PROJECT_ROOT"

# Define an array of model names.
worker_model_list=(
    "unsloth_qwen_0.5B"
    "unsloth_qwen3_0.6B"
    "unsloth_llama_1B-Instruct"
    "unsloth_deepseek_r1_qwen_1.5B"
    "unsloth_qwen3_1.7B"
    #"unsloth_llama_3B-Instruct"
    #"unsloth_qwen_3B"
    #"unsloth_qwen3_4B"
    #"qwen3_30B_A3B"
)

#
#refiner_model_list=(
#    "unsloth_qwen_0.5B"
#    "unsloth_qwen3_0.6B"
#    "unsloth_llama_1B-Instruct"
#    "unsloth_deepseek_r1_qwen_1.5B"
#    "unsloth_qwen3_1.7B"
#    "unsloth_llama_3B-Instruct"
#    "unsloth_qwen_3B"
#    "unsloth_qwen3_4B"
#)

DATASETS=(
    #"adult"
    "titanic"
    "california"
    "diabetes"
)

# Iterate over each model configuration and run experiment.py accordingly.
for dataset in "${DATASETS[@]}"; do
    for worker_model in "${worker_model_list[@]}"; do
    echo "⚙️ Running experiment.py with worker model '$worker_model'..."
    
    # Use this line to run asymmetric experiments (worker model not equal to refiner model)
    #for refiner_model in "${refiner_model_list[@]}"; do
        #echo "⚙️⚙️ Using refiner model '$refiner_model'..."
        # python experiment.py --dataset="adult" --worker_model_name="$worker_model" --refiner_model_name="$refiner_model" --refiner --worker_fine_tuned --analyze_feasibility
        #python experiment.py --dataset="adult" --worker_model_name="$worker_model" --refiner_model_name="$refiner_model" --refiner --worker_fine_tuned --refiner_fine_tuned --analyze_feasibility
        
        python experiment.py --dataset="adult" --worker_model_name="$worker_model" --refiner_model_name="$worker_model" --refiner --worker_fine_tuned --analyze_feasibility
        python experiment.py --dataset="adult" --worker_model_name="$worker_model" --refiner_model_name="$worker_model" --refiner --worker_fine_tuned --refiner_fine_tuned --analyze_feasibility
    done
done

