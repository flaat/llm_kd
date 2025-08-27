# Define an array of model names.
model_list=(
    "unsloth_qwen_0.5B"
    "unsloth_deepseek_r1_qwen_1.5B"
    "unsloth_qwen_3B"
    "unsloth_deepseek_r1_qwen_7B",
    #"deepseek_r1_qwen_32B_Q4_AWQ1"

)

# Iterate over each model configuration and run main.py accordingly.
for model in "${model_list[@]}"; do
    echo "Running main.py with model '$model'..."
    python main.py  --dataset=adult --worker_model_name="$model" --test_llm --fine_tuned 
done