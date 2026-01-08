"""
Model Fine-tuning Script using Unsloth and HuggingFace

This script fine-tunes large language models using LoRA (Low-Rank Adaptation) technique
through Unsloth's optimization wrapper around HuggingFace's transformers. It:

1. Loads a pre-trained model from a configurable mapping
2. Applies LoRA configuration for efficient fine-tuning
3. Loads training data from a JSON file
4. Formats the data for instruction-based training
5. Trains the model using SFTTrainer
6. Reports memory usage statistics and training time

The script supports various model types and configurations through command-line arguments.

Example usage:
    python tests.py --model_name phi_4B
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import time
import subprocess
import threading

from unsloth import FastLanguageModel, is_bfloat16_supported, get_chat_template
from unsloth.chat_templates import train_on_responses_only
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# Assumes src.utils exists in your project structure
from src.utils import MODEL_MAPPING

# Constants
MAX_SEQ_LENGTH = 8000  # Context window size
DTYPE = None  # None for auto detection: Float16 for Tesla T4/V100, BFloat16 for Ampere+
LOAD_IN_4BIT = True  # Use 4-bit quantization to reduce memory usage

def load_model_and_tokenizer(model_name: str) -> Tuple[Any, Any]:
    """
    Load a pre-trained model and tokenizer using Unsloth's optimized loader.
    
    Args:
        model_name: The name key for the model in MODEL_MAPPING
    
    Returns:
        Tuple containing (model, tokenizer)
    """
    # Load base model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_MAPPING[model_name],
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Configure for LoRA fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank - recommended values: 8, 16, 32, 64, 128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # 0 is optimized
        bias="none",  # "none" is optimized
        use_gradient_checkpointing="unsloth",  # Uses less VRAM
        random_state=3407,
        use_rslora=False,  # Rank stabilized LoRA
        loftq_config=None,  # LoftQ configuration
    )
    
    return model, tokenizer


def load_json_to_hf_dataset(json_path: str) -> Dataset:
    """
    Load a JSON file and convert it into a Hugging Face dataset.

    Args:
        json_path: Path to the JSON file containing prompts and responses

    Returns:
        A Hugging Face dataset object with 'prompt' and 'response' fields
    """
    # Load JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract data entries
    dataset_entries = []
    for _, value in data.items():
        dataset_entries.append({
            "prompt": value.get("prompt", ""),  # Extract prompt
            "response": value.get("generated_text", "")  # Extract model response
        })

    # Convert to Hugging Face Dataset
    return Dataset.from_list(dataset_entries)


def create_formatting_function(tokenizer):
    """
    Create a function to format examples for chat-based instruction fine-tuning.
    
    Args:
        tokenizer: The tokenizer to use for formatting
        
    Returns:
        A function that formats dataset examples into model inputs
    """
    def formatting_prompts_func(example):
        instruction = example["prompt"]
        output = example["response"]

        # Format as a chat template while preserving special tokens
        # We use a trick to preserve <think></think> tags by temporarily replacing them
        tokenizer_input = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output.replace("<think>", "<><><>think<><><>")
                                                  .replace("</think>", "<><><>/think<><><>")},
        ]
        
        # Apply chat template and restore special tokens
        text = tokenizer.apply_chat_template(tokenizer_input, tokenize=False)
        text = text.replace("<><><>think<><><>", "<think>").replace("<><><>/think<><><>", "</think>")
        
        return {"text": text}
    
    return formatting_prompts_func


def setup_trainer(model, tokenizer, dataset, output_dir: str) -> SFTTrainer:
    """
    Configure and create the SFT trainer for fine-tuning.
    Compatible with trl >= 0.24.0.
    """

    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",          
        max_seq_length=MAX_SEQ_LENGTH,      
        dataset_num_proc=1,                 
        packing=False,                      
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        warmup_ratio=0.075,
        num_train_epochs=1,
        learning_rate=2e-4,
        overwrite_output_dir=False,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",                   # Set to "wandb" for Weights & Biases logging
        save_strategy="steps",
        save_steps=50
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,                    # Pass the SFTConfig
        processing_class=tokenizer,         # UPDATED: 'tokenizer' arg is removed in 0.24, use 'processing_class'
    )

    return trainer


def get_response_only_kwargs(model_name: str) -> Optional[Dict[str, str]]:
    """
    Return instruction/response delimiters so loss is computed only on assistant
    replies. The spans must match the chat template emitted by the tokenizer.
    """
    name = model_name.lower()

    # Qwen2.5 / Qwen3 (unsloth) use <|im_start|> markers.
    if "qwen" in name:
        return {
            "instruction_part": "<|im_start|>user",
            "response_part": "<|im_start|>assistant",
        }

    # Llama 3.2 uses start/end header ids around roles.
    if "llama" in name:
        return {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>",
        }

    return None


def print_memory_stats(start_gpu_memory: float, trainer_stats: Optional[Any] = None):
    """
    Print GPU memory usage statistics.
    
    Args:
        start_gpu_memory: Starting GPU memory usage in GB
        trainer_stats: Training statistics from trainer.train() (if available)
    """
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    # Initial memory stats
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    # Print final stats if training is complete
    if trainer_stats:
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


class EnergyMonitor:
    """Monitor energy consumption during fine-tuning using nvidia-smi."""
    
    def __init__(self):
        self.monitoring = False
        self.power_samples = []  # Store (timestamp, power) tuples
    
    def get_gpu_power_draw(self):
        """Get current GPU power draw in watts using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2.0
            )
            if result.returncode == 0:
                power_draws = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            power_draws.append(float(line.strip()))
                        except ValueError:
                            continue
                return sum(power_draws) if power_draws else 0.0
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            print(f"⚠️ nvidia-smi failed: {e}")
        return 0.0
    
    def start_monitoring(self):
        """Start energy monitoring with precise timestamps."""
        self.monitoring = True
        self.power_samples = []
        
        def monitor():
            while self.monitoring:
                try:
                    timestamp = time.time()
                    power = self.get_gpu_power_draw()
                    # Store all power readings, including zeros
                    self.power_samples.append((timestamp, power))
                    # Sleep for 200ms (nvidia-smi is slower than NVML)
                    time.sleep(0.2)
                except Exception as e:
                    print(f"⚠️ Monitoring error: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop energy monitoring and return detailed energy metrics."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=3.0)
        
        if len(self.power_samples) < 2:
            return {
                "energy_joules": 0.0,
                "average_power_watts": 0.0,
                "duration_seconds": 0.0,
                "sample_count": 0,
            }
        
        # Calculate energy using trapezoidal integration
        total_energy = 0.0
        total_power = 0.0
        
        for i in range(1, len(self.power_samples)):
            prev_time, prev_power = self.power_samples[i - 1]
            curr_time, curr_power = self.power_samples[i]
            
            # Time difference between samples
            dt = curr_time - prev_time
            
            # Trapezoidal integration: area = (p1 + p2) * dt / 2
            energy_segment = (prev_power + curr_power) * dt / 2.0
            total_energy += energy_segment
            total_power += curr_power
        
        # Calculate metrics
        duration = self.power_samples[-1][0] - self.power_samples[0][0]
        average_power = total_power / (len(self.power_samples) - 1) if len(self.power_samples) > 1 else 0.0
        
        return {
            "energy_joules": total_energy,
            "average_power_watts": average_power,
            "duration_seconds": duration,
            "sample_count": len(self.power_samples),
        }
    
    def cleanup(self):
        """Cleanup resources (no-op for nvidia-smi version)."""
        pass


def main(model_name: str, dataset_name: str, refiner: bool) -> None:
    """
    Main function to execute the fine-tuning pipeline.
    
    Args:
        model_name: Name of the model to fine-tune (key in MODEL_MAPPING)
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Attach the proper chat template to the tokenizer
    if "qwen" in model_name:
        tokenizer = get_chat_template(tokenizer, chat_template="qwen25")
    elif "llama" in model_name:
        tokenizer = get_chat_template(tokenizer, chat_template="llama-31")
    
    # Load dataset
    if refiner:
        json_file_path = f"data/{dataset_name}_refiner_unsloth_qwen3_1.7B--qwen3_30B_A3B_cleaned.json"
    else:
        json_file_path = f"data/{dataset_name}_worker_qwen3_30B_A3B_cleaned.json"
    
    print(json_file_path)
    dataset = load_json_to_hf_dataset(json_file_path)
    
    # Create formatting function and process dataset
    formatting_func = create_formatting_function(tokenizer)
    formatted_dataset = dataset.map(formatting_func, batched=False)
    #Print the first example of the formatted dataset
    print(formatted_dataset[0])
    
    # Prepare output directory

    if refiner:
        output_dir = f"outputs_unsloth_{dataset_name}_refiner/{model_name}"
    else:
        output_dir = f"outputs_unsloth_{dataset_name}_worker/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up trainer
    trainer = setup_trainer(model, tokenizer, formatted_dataset, output_dir)

    # Mask loss to assistant responses only (prevent user prompt tokens from
    # contributing to loss). Defaults to full-text loss when no mapping exists.
    response_only_kwargs = get_response_only_kwargs(model_name)
    if response_only_kwargs:
        trainer = train_on_responses_only(trainer, **response_only_kwargs)
    
    # Record initial memory usage
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print_memory_stats(start_gpu_memory)

    # Prepare feasibility output path
    os.makedirs("results/feasibility", exist_ok=True)
    role = "refiner" if refiner else "worker"
    feasibility_output_file = f"results/feasibility/{dataset_name}_{role}_finetune_{model_name}.json"

    # Initialize energy monitor for fine-tuning
    energy_monitor = EnergyMonitor()

    # Start timing and energy monitoring for the full fine-tuning run
    energy_monitor.start_monitoring()
    start_time = time.time()

    # Train the model
    trainer_stats = trainer.train()

    # Stop timing and energy monitoring
    end_time = time.time()
    energy_metrics = energy_monitor.stop_monitoring()

    # Print final memory statistics
    print_memory_stats(start_gpu_memory, trainer_stats)

    # Compute overall fine-tuning time
    total_time_seconds = end_time - start_time

    # Build feasibility statistics dictionary
    feasibility_stats = {
        "total_time_seconds": total_time_seconds,
        "total_energy_joules": energy_metrics["energy_joules"],
        "average_power_watts": energy_metrics["average_power_watts"],
        "duration_seconds": energy_metrics["duration_seconds"],
        "sample_count": energy_metrics["sample_count"],
        "model_name": model_name,
        "dataset_name": dataset_name,
        "refiner": refiner,
    }

    # Save feasibility statistics
    with open(feasibility_output_file, "w", encoding="utf-8") as f:
        json.dump(feasibility_stats, f, indent=4)
    print(f"✅ Feasibility statistics saved to {feasibility_output_file}")

    # Cleanup energy monitor resources
    energy_monitor.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune language models with LoRA.")
    parser.add_argument(
        '--model_name',
        type=str,
        default='phi_4B',
        help='Model name key from MODEL_MAPPING (default: phi_4B)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='titanic',
        help='Dataset name to use for fine-tuning (default: titanic)'
    )
    parser.add_argument(
        '--refiner',
        action='store_true',
        help='Use this flag to indicate that the dataset is a refiner dataset (default: False)'
    )
    args = parser.parse_args()
    main(args.model_name, args.dataset, args.refiner)