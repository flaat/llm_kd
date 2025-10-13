# Define an array of model names.
worker_model_list=(
    "unsloth_qwen_0.5B"
    "unsloth_deepseek_r1_qwen_1.5B"
    "unsloth_qwen_3B"
    "unsloth_deepseek_r1_qwen_7B"
)
refiner_model_list=(
    "unsloth_qwen_0.5B"
    "unsloth_deepseek_r1_qwen_1.5B"
    "unsloth_qwen_3B"
    "unsloth_deepseek_r1_qwen_7B"
)

# Iterate over each model configuration and run main.py accordingly.
for worker_model in "${worker_model_list[@]}"; do
    echo "⚙️ Running main.py with worker model '$worker_model'..."
    for refiner_model in "${refiner_model_list[@]}"; do
        echo "⚙️⚙️ Using refiner model '$refiner_model'..."
        python main.py --test_llm --dataset="adult" --worker_model_name="$worker_model" --refiner_model_name="$refiner_model" --refiner --fine_tuned 
    done
done