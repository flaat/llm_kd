# Define an array of model names.
model_list=(
    "unsloth_qwen_0.5B"
    "unsloth_deepseek_r1_qwen_1.5B"
    "unsloth_qwen_3B"
    "unsloth_deepseek_r1_qwen_7B"
)

# Iterate over each model configuration and run tests.py accordingly.
for model in "${model_list[@]}"; do
    echo "Running tests.py with model '$model'..."
    python main.py --test_llm=True --model_name="$model" --fine_tuned=True
done