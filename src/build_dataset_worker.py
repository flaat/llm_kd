from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from data.dataset_kb import dataset_kb
import random
import numpy as np
from .utils import MODEL_MAPPING, prompt, prompt_ref
import json
import time
import os
import re
import subprocess
import threading

def set_full_reproducibility(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_existing_responses(output_file):
    """Loads existing responses if the file already exists."""
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                print("⚠️ Warning: Existing file is corrupted. Starting fresh.")
                return {}
    return {}

def save_responses(responses, output_file):
    """ Appends new responses to the existing JSON file. """
    existing_responses = load_existing_responses(output_file)
    
    # Ensure unique indexing when appending new responses
    offset = max(map(int, existing_responses.keys()), default=-1) + 1
    for i, response in enumerate(responses.values(), start=offset):
        existing_responses[str(i)] = response  # Store keys as strings to match JSON formatting
    
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(existing_responses, output, indent=4)
    print(f"✅ Responses saved to {output_file}.")


class EnergyMonitor:
    """Monitor energy consumption during inference using nvidia-smi."""
    
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
                'energy_joules': 0.0,
                'average_power_watts': 0.0,
                'duration_seconds': 0.0,
                'sample_count': 0
            }
        
        # Calculate energy using trapezoidal integration
        total_energy = 0.0
        total_power = 0.0
        
        for i in range(1, len(self.power_samples)):
            prev_time, prev_power = self.power_samples[i-1]
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
            'energy_joules': total_energy,
            'average_power_watts': average_power,
            'duration_seconds': duration,
            'sample_count': len(self.power_samples)
        }
    
    def cleanup(self):
        """Cleanup resources (no-op for nvidia-smi version)."""
        pass


def build_dataset_wor(model_name: str, temperature: float, top_p: float, dataset: str, max_tokens: int, repetition_penalty: float, max_model_len, analyze_feasibility: bool = True):
    set_full_reproducibility()
    
    LOWER_BOUND = 0
    UPPER_BOUND = 50000
    
    global prompt
    base_prompt = prompt
    
    # Map model names to HuggingFace identifiers
    model_path = MODEL_MAPPING[model_name]

    # Initialize tokenizers for both models
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        repetition_penalty=repetition_penalty, 
        max_tokens=max_tokens, 
        top_k=10
    )

    print(f"📋 Model: {model_name}")
    # Load worker model
    print(f"🔧 Loading Worker Model: {model_name}")
    llm = LLM(
        model=model_path, 
        gpu_memory_utilization=0.92, 
        max_model_len=max_model_len, 
        max_num_seqs=1
    )

    # Define output files
    # Responses are always saved to data/ folder
    responses_output_file = f"data/{dataset}_Worker_{model_name}.json"
    # Feasibility statistics are saved to results/feasibility/ folder when needed
    feasibility_output_file = f"results/feasibility/{dataset}_worker_{model_name}.json"
    
    # Ensure results/feasibility directory exists
    os.makedirs("results/feasibility", exist_ok=True)
    
    responses = {}  # Dictionary to store new responses
    
    # Initialize feasibility metrics if analyzing feasibility
    # Use incremental statistics to avoid storing all values
    if analyze_feasibility:
        feasibility = {
            "total_time_seconds": 0.0,
            "total_energy_joules": 0.0,
            "num_generations": 0,
            # For variance calculation (Welford's online algorithm)
            "time_M2": 0.0,  # Sum of squared differences for time
            "energy_M2": 0.0,  # Sum of squared differences for energy
            "time_mean": 0.0,
            "energy_mean": 0.0
        }

    # Load counterfactual data
    with open(f"src/explainer/counterfactuals.json", 'r', encoding='utf-8') as file1:
        data = json.load(file1)

    
    i = 0  # Counter for responses

    if dataset == "adult":
        dataset = "adult income"
    elif dataset == "california":
        dataset = "california housing"

    for dataset_name, examples in data.items():

        if dataset_name.lower() == dataset:
            for index, values in examples.items():
                for counterfactual in values["counterfactuals"]:
                    
                    if LOWER_BOUND <= i <= UPPER_BOUND: 
                        
                        current_prompt = base_prompt.format(
                            dataset_description=dataset_kb[dataset_name], 
                            factual_example=str(values["factual"]), 
                            counterfactual_example=str(counterfactual)
                        )

                        messages = [{"role": "user", "content": current_prompt}]

                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        # Initialize energy monitor
                        energy_monitor = EnergyMonitor()
                        
                        # Start timing and energy monitoring
                        energy_monitor.start_monitoring()
                        start = time.time()
                        
                        try:
                            with torch.no_grad():
                                outputs = llm.generate([text], sampling_params=sampling_params)
                        except AssertionError as assert_e:
                            print(f"🚨 Assertion error: {assert_e}")
                            energy_monitor.stop_monitoring()
                            continue
                        
                        # Stop timing and energy monitoring
                        end = time.time()
                        energy_monitor.stop_monitoring()
                        energy_metrics = energy_monitor.stop_monitoring()
                        
                        # Calculate metrics
                        inference_time = end - start
                        energy_consumed = energy_metrics['energy_joules']
                        
                        # Process the outputs if generated successfully
                        for output in outputs:
                            prompt = output.prompt
                            generated_text = output.outputs[0].text

                        print(generated_text)
                        
                        # Always store response data
                        responses[i] = {"prompt": prompt, "generated_text": generated_text}
                        
                        # Update statistics incrementally if analyzing feasibility
                        if analyze_feasibility:
                            # Update counters
                            feasibility["num_generations"] += 1
                            n = feasibility["num_generations"]
                            
                            # Update totals
                            feasibility["total_time_seconds"] += inference_time
                            feasibility["total_energy_joules"] += energy_consumed
                            
                            # Welford's online algorithm for variance (time)
                            delta_time = inference_time - feasibility["time_mean"]
                            feasibility["time_mean"] += delta_time / n
                            delta_time2 = inference_time - feasibility["time_mean"]
                            feasibility["time_M2"] += delta_time * delta_time2
                            
                            # Welford's online algorithm for variance (energy)
                            delta_energy = energy_consumed - feasibility["energy_mean"]
                            feasibility["energy_mean"] += delta_energy / n
                            delta_energy2 = energy_consumed - feasibility["energy_mean"]
                            feasibility["energy_M2"] += delta_energy * delta_energy2

                        print(f"#################### explanation #{i} - Time taken: {inference_time:.2f}s, Energy: {energy_consumed:.2f}J ###########################")

                        # Save responses every 10 iterations
                        if i % 10 == 0:
                            save_responses(responses, responses_output_file)
                            responses = {}  # Clear the temporary dictionary to prevent duplication
                        
                        # Save feasibility stats periodically (every 100 generations) to track progress
                        if analyze_feasibility and i % 100 == 0 and feasibility["num_generations"] > 0:
                            avg_time = feasibility["total_time_seconds"] / feasibility["num_generations"]
                            avg_energy = feasibility["total_energy_joules"] / feasibility["num_generations"]
                            print(f"📊 Progress: {feasibility['num_generations']} generations - Avg time: {avg_time:.2f}s, Avg energy: {avg_energy:.2f}J")

                        # Delete large variables to free memory
                        del generated_text
                        del outputs
                    i += 1
    
    # Final save of responses after loop completion
    save_responses(responses, responses_output_file)
    
    # Save feasibility statistics if analyzing feasibility
    if analyze_feasibility:
        # Calculate final statistics from incremental data
        if feasibility["num_generations"] > 0:
            n = feasibility["num_generations"]
            
            # Calculate standard deviations
            time_variance = feasibility["time_M2"] / n if n > 1 else 0.0
            energy_variance = feasibility["energy_M2"] / n if n > 1 else 0.0
            time_std = np.sqrt(time_variance)
            energy_std = np.sqrt(energy_variance)
            
            # Create statistics dictionary
            stats = {
                "total_time_seconds": feasibility["total_time_seconds"],
                "total_energy_joules": feasibility["total_energy_joules"],
                "average_inference_time_seconds": feasibility["total_time_seconds"] / n,
                "average_energy_per_generation_joules": feasibility["total_energy_joules"] / n,
                "std_inference_time_seconds": time_std,
                "std_energy_per_generation_joules": energy_std,
                "num_generations": n
            }
            
            # Save feasibility statistics
            with open(feasibility_output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4)
            print(f"✅ Feasibility statistics saved to {feasibility_output_file}")
            
            # Print summary
            print(f"\n📊 Feasibility Summary:")
            print(f"Total time: {stats['total_time_seconds']:.2f} seconds")
            print(f"Total energy: {stats['total_energy_joules']:.2f} Joules")
            print(f"Average inference time per generation: {stats['average_inference_time_seconds']:.2f} ± {stats['std_inference_time_seconds']:.2f} seconds")
            print(f"Average energy per generation: {stats['average_energy_per_generation_joules']:.2f} ± {stats['std_energy_per_generation_joules']:.2f} Joules")
            print(f"Number of generations: {stats['num_generations']}")
    
    # Cleanup energy monitor resources
    if 'energy_monitor' in locals():
        energy_monitor.cleanup()

