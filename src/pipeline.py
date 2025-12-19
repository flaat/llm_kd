import logging
import io
from transformers import AutoTokenizer
import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import random
import numpy as np
import time
import os
import subprocess
import threading
from data.dataset_kb import dataset_kb
from .utils import MODEL_MAPPING, prompt, prompt_ref, get_checkpoint_step
import re
from vllm.lora.request import LoRARequest

def set_full_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
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

def extract_feature_changes(factual, counterfactual):
    """
    Extracts feature changes between factual and counterfactual examples.

    Args:
        ground_truth (dict): A dictionary containing 'factual' and 'counterfactual' samples.

    Returns:
        dict: A dictionary containing only the features that changed, formatted as specified.
    """


    feature_changes = []

    for key in factual:
        if key in counterfactual and factual[key] != counterfactual[key]:
            feature_changes.append({
                key: {
                    "factual": factual[key],
                    "counterfactual": counterfactual[key]
                }
            })

    return {"feature_changes": feature_changes}

def extract_explanations(results: list[str]):
    """
    Extracts explanations from the generated outputs.
    """
    explanations = []
    for outputs in results:

        for output in outputs:
            text = output.outputs[0].text

        if not text:
            print("⚠️ No text generated in the output.")

        try:
            # Attempt to extract JSON block within triple backticks
            json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
            if not json_match:
                # Attempt to extract JSON directly enclosed in curly brackets
                json_match = re.search(r"({.*})", text, re.DOTALL)

            if json_match:
                json_string = json_match.group(1).strip()
                response = json.loads(json_string)
                try :
                    explanation = response["explanation"]
                    explanations.append(explanation)
                except KeyError:
                    print(f"⚠️ 'explanation' key not found in the response: {response}")
            else:
                print(f"⚠️ No JSON block found in the given text: {text}")

        except json.JSONDecodeError:
            print(f"⚠️ JSON parsing error in {text}")

    return tuple(explanations[:3] + [None] * (3 - len(explanations)))


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
            'power_samples': self.power_samples,
            'duration_seconds': duration,
            'sample_count': len(self.power_samples)
        }
    
    def cleanup(self):
        """Cleanup resources (no-op for nvidia-smi version)."""
        pass

class VLLMLogCapture:
    """Capture vLLM logs to extract model loading information."""
    
    def __init__(self):
        self.model_weight_size_gb = 0.0
        self.log_output = []
        
    def capture_logs(self):
        """Setup log capture for vLLM."""
        # Create a custom handler to capture logs
        self.log_stream = io.StringIO()
        
        # Get the vLLM logger
        vllm_logger = logging.getLogger('vllm')
        
        # Create handler that writes to our string buffer
        handler = logging.StreamHandler(self.log_stream)
        handler.setLevel(logging.INFO)
        
        # Add our handler
        vllm_logger.addHandler(handler)
        vllm_logger.setLevel(logging.INFO)
        
        return handler
    
    def extract_model_size(self, handler):
        """Extract model size from captured logs."""
        # Remove our handler
        vllm_logger = logging.getLogger('vllm')
        vllm_logger.removeHandler(handler)
        
        # Get the log content
        log_content = self.log_stream.getvalue()
        
        # Look for the pattern "Loading model weights took X.XXXX GB"
        import re
        pattern = r"Loading model weights took ([\d.]+) GB"
        match = re.search(pattern, log_content)
        
        if match:
            self.model_weight_size_gb = float(match.group(1))
            print(f"📊 Captured model weight size: {self.model_weight_size_gb:.4f} GB")
        else:
            print("⚠️ Could not capture model weight size from logs")
            # Fallback to 0
            self.model_weight_size_gb = 0.0
            
        return self.model_weight_size_gb



def get_gpu_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    return 0


def test_llm(
    worker_model_name: str,
    refiner_model_name: str | None,
    dataset: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    max_model_len,
    refiner: bool = False,
    worker_fine_tuned: bool = False,
    refiner_fine_tuned: bool = False,
    analyze_feasibility: bool = False,
    num_narratives: int = 3,
):
    """
    Unified entrypoint for running LLM experiments.

    If `refiner` is False, runs the single-model pipeline (former `test_llm`)
    using `worker_model_name`.

    If `refiner` is True, runs the worker+refiner pipeline (former
    `test_llm_refiner`) using both `worker_model_name` and `refiner_model_name`.
    """

    print(
        "[CONFIG] Params: "
        f"worker_model={worker_model_name}, refiner_model={refiner_model_name}, "
        f"dataset={dataset}, temperature={temperature}, top_p={top_p}, "
        f"max_tokens={max_tokens}, repetition_penalty={repetition_penalty}, "
        f"max_model_len={max_model_len}, "
        f"worker_fine_tuned={worker_fine_tuned}, refiner_fine_tuned={refiner_fine_tuned}, "
        f"refiner={refiner}, analyze_feasibility={analyze_feasibility}"
    )
    set_full_reproducibility()

    if not refiner:
        _run_worker_only_pipeline(
            worker_model_name=worker_model_name,
            dataset=dataset,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            max_model_len=max_model_len,
            worker_fine_tuned=worker_fine_tuned,
            analyze_feasibility=analyze_feasibility,
        )
    else:
        if refiner_model_name is None:
            refiner_model_name = worker_model_name

        _run_worker_refiner_pipeline(
            worker_model_name=worker_model_name,
            refiner_model_name=refiner_model_name,
            dataset=dataset,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            max_model_len=max_model_len,
            worker_fine_tuned=worker_fine_tuned,
            refiner_fine_tuned=refiner_fine_tuned,
            analyze_feasibility=analyze_feasibility,
            num_narratives=num_narratives,
        )


def _run_worker_only_pipeline(
    worker_model_name: str,
    dataset: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    max_model_len,
    worker_fine_tuned: bool,
    analyze_feasibility: bool,
):
    """Former `test_llm` pipeline (single model, no refiner)."""

    LOWER_BOUND = 1
    UPPER_BOUND = 200
    logical_name = worker_model_name
    global prompt
    base_prompt = prompt
    mapped_model_name = MODEL_MAPPING[worker_model_name]

    print(f"[WORKER] Loading tokenizer and model '{logical_name}' ({mapped_model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(mapped_model_name)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        top_k=10,
        stop=tokenizer.eos_token,
    )

    # Calculate model disk size using vLLM log capture
    log_capture = VLLMLogCapture()

    # Measure model loading time
    print("[WORKER] 🔄 Loading model...")
    model_load_start = time.time()

    # Setup log capture before model loading
    handler = log_capture.capture_logs()

    # Initialize LLM with optimized GPU memory usage
    llm = LLM(
        model=mapped_model_name,
        gpu_memory_utilization=0.6,
        max_model_len=max_model_len,
        max_num_seqs=1,
        enable_lora=worker_fine_tuned,
    )

    model_load_end = time.time()
    model_loading_time = model_load_end - model_load_start

    # Extract model size from logs
    model_disk_size_gb = log_capture.extract_model_size(handler)

    print(f"[WORKER] 📊 Model loading time: {model_loading_time:.2f} seconds")
    print(f"[WORKER] 📊 Model weight size: {model_disk_size_gb:.4f} GB")

    # Get GPU memory usage after model loading
    gpu_memory_usage_gb = get_gpu_memory_usage()
    print(f"[WORKER] 📊 GPU memory usage after loading: {gpu_memory_usage_gb:.2f} GB")

    # LoRA checkpoint (only used if worker_fine_tuned is True)
    checkpoint_step = get_checkpoint_step(dataset, "draft_generator", logical_name, default=500)
    lora_checkpoint_directory_path = (
        f"outputs_unsloth/outputs_unsloth_{dataset}_worker/{logical_name}/checkpoint-{checkpoint_step}"
    )
    print(
        f"[CHECKPOINT] Worker draft generator checkpoint: "
        f"dataset={dataset}, model={logical_name}, step={checkpoint_step}"
    )

    # Define output file for results
    responses = {}  # Dictionary to store new responses
    feasibility = {  # Dictionary to store feasibility results
        "loading_time": model_loading_time,
        "model_weight_size_gb": model_disk_size_gb,
        "gpu_memory_usage_gb": gpu_memory_usage_gb,
        "inference_time": [],
        "energy_consumption": [],
        "average_power": [],
        "power_samples": [],
    }

    # Load counterfactual data
    with open("src/explainer/test_counterfactuals.json", "r", encoding="utf-8") as file1:
        data1 = json.load(file1)

    i = 0  # Counter for responses
    NUM_FACTUALS = 40  # Number of factuals to process
    NUM_COUNTERFACTUALS_PER_FACTUAL = 5  # Number of counterfactuals per factual
    if NUM_FACTUALS * NUM_COUNTERFACTUALS_PER_FACTUAL != UPPER_BOUND:
        raise ValueError(f"NUM_FACTUALS * NUM_COUNTERFACTUALS_PER_FACTUAL must be {UPPER_BOUND}")

    for dataset_name, examples in data1.items():
        dataset_key = dataset
        if dataset_key == "adult":
            dataset_key = "adult income"
        elif dataset_key == "california":
            dataset_key = "california housing"

        if dataset_name.lower() == dataset_key:
            # Ensure output directory structure
            base_dir = os.path.join(
                "results",
                "draft_generator",
                dataset_name,
                logical_name,
            )
            os.makedirs(base_dir, exist_ok=True)

            # Define output files
            if not analyze_feasibility:
                output_file = os.path.join(
                    base_dir,
                    f"{logical_name}_response_finetuned_{worker_fine_tuned}.json",
                )
                feasibility_output_file = None
            else:
                # When analyzing feasibility, save both test results and feasibility metrics
                output_file = os.path.join(
                    base_dir,
                    f"{logical_name}_response_finetuned_{worker_fine_tuned}.json",
                )
                feasibility_dir = os.path.join(
                    base_dir,
                    "feasibility",
                )
                os.makedirs(feasibility_dir, exist_ok=True)
                feasibility_output_file = os.path.join(
                    feasibility_dir,
                    f"{logical_name}_feasibility_response_finetuned_{worker_fine_tuned}.json",
                )

            print(f"[IO] Test results output file: {output_file}")
            if feasibility_output_file:
                print(f"[IO] Feasibility metrics output file: {feasibility_output_file}")
            print(f"[CONFIG] Processing {NUM_FACTUALS} factuals with {NUM_COUNTERFACTUALS_PER_FACTUAL} counterfactuals each (total: {NUM_FACTUALS * NUM_COUNTERFACTUALS_PER_FACTUAL} iterations)")

            # Sort indices to ensure consistent ordering (indices are strings like "0", "1", etc.)
            sorted_indices = sorted(examples.keys(), key=lambda x: int(x))[:NUM_FACTUALS]
            
            for index in sorted_indices:
                values = examples[index]
                # Process only the first NUM_COUNTERFACTUALS_PER_FACTUAL counterfactuals
                counterfactuals_to_process = values["counterfactuals"][:NUM_COUNTERFACTUALS_PER_FACTUAL]
                
                for counterfactual in counterfactuals_to_process:
                    if LOWER_BOUND <= i <= UPPER_BOUND:
                        current_prompt = base_prompt.format(
                            dataset_description=dataset_kb[dataset_name],
                            factual_example=str(values["factual"]),
                            counterfactual_example=str(counterfactual),
                        )

                        messages = [{"role": "user", "content": current_prompt}]

                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        try:
                            energy_monitor = EnergyMonitor()

                            with torch.no_grad():
                                # Start energy monitoring and time measurement
                                energy_monitor.start_monitoring()
                                start = time.time()

                                if worker_fine_tuned:
                                    outputs = llm.generate(
                                        [text],
                                        sampling_params=sampling_params,
                                        lora_request=LoRARequest(
                                            "counterfactual_explainability_adapter",
                                            1,
                                            lora_checkpoint_directory_path,
                                        ),
                                    )
                                else:
                                    outputs = llm.generate([text], sampling_params=sampling_params)

                                end = time.time()
                                energy_metrics = energy_monitor.stop_monitoring()

                        except AssertionError as assert_e:
                            print(f"[WORKER] 🚨 Assertion error: {assert_e}")
                            continue

                        # Calculate metrics
                        inference_time = end - start
                        energy_consumed = energy_metrics["energy_joules"]
                        average_power = energy_metrics["average_power_watts"]
                        power_samples_tot = energy_metrics["power_samples"]
                        power_samples = [sample[1] for sample in power_samples_tot]

                        # Process the outputs if generated successfully
                        for output in outputs:
                            prompt_out = output.prompt
                            generated_text = output.outputs[0].text

                        print(f"[WORKER] Generated explanation #{i}:")
                        print(generated_text)

                        # Store response with metrics
                        response_data = {
                            "generated_text": generated_text,
                            "prompt": prompt_out,
                            "ground_truth": {
                                "counterfactual": counterfactual,
                                "factual": values["factual"],
                            },
                            "changes": extract_feature_changes(values["factual"], counterfactual),
                        }
                        responses[i] = response_data

                        # Store metrics in feasibility dictionary if analyzing feasibility
                        if analyze_feasibility:
                            feasibility["inference_time"].append(inference_time)
                            feasibility["energy_consumption"].append(energy_consumed)
                            feasibility["average_power"].append(average_power)
                            feasibility["power_samples"].append(power_samples)

                        print(
                            f"[WORKER] #################### explanation #{i} - "
                            f"Time: {inference_time:.2f}s, Energy: {energy_consumed:.2f}J, "
                            f"Avg Power: {average_power:.2f}W ###########################"
                        )

                        # Save every 10 responses (always save test results)
                        if i % 10 == 0:
                            save_responses(responses, output_file)
                            responses = {}  # Clear the temporary dictionary to prevent duplication

                        # Delete large variables to free memory
                        del generated_text
                        del outputs
                    i += 1

    # Final save after loop completion
    # Always save test results
    save_responses(responses, output_file)
    
    # If analyzing feasibility, also save feasibility metrics to separate file
    if analyze_feasibility:
        # Calculate statistics before saving
        if feasibility["inference_time"]:
            feasibility["inference_time_stats"] = {
                "mean": float(np.mean(feasibility["inference_time"])),
                "std": float(np.std(feasibility["inference_time"])),
                "min": float(np.min(feasibility["inference_time"])),
                "max": float(np.max(feasibility["inference_time"])),
            }

        if feasibility["energy_consumption"]:
            feasibility["energy_consumption_stats"] = {
                "mean": float(np.mean(feasibility["energy_consumption"])),
                "std": float(np.std(feasibility["energy_consumption"])),
                "min": float(np.min(feasibility["energy_consumption"])),
                "max": float(np.max(feasibility["energy_consumption"])),
            }

        if feasibility["average_power"]:
            feasibility["average_power_stats"] = {
                "mean": float(np.mean(feasibility["average_power"])),
                "std": float(np.std(feasibility["average_power"])),
                "min": float(np.min(feasibility["average_power"])),
                "max": float(np.max(feasibility["average_power"])),
            }

        # Save feasibility metrics to separate file
        with open(feasibility_output_file, "w", encoding="utf-8") as f:
            json.dump(feasibility, f, indent=4)
        print(f"[FEASIBILITY] ✅ Metrics saved to {feasibility_output_file}")

        if feasibility["inference_time"]:
            print("\n[FEASIBILITY] 📊 Summary:")
            print(f" - Model loading time: {model_loading_time:.2f} seconds")
            print(f" - Model weight size: {model_disk_size_gb:.4f} GB")
            print(f" - GPU memory usage: {gpu_memory_usage_gb:.2f} GB")
            print(
                " - Average inference time: "
                f"{np.mean(feasibility['inference_time']):.2f} "
                f"± {np.std(feasibility['inference_time']):.2f} seconds"
            )
            print(
                " - Average energy consumption: "
                f"{np.mean(feasibility['energy_consumption']):.2f} "
                f"± {np.std(feasibility['energy_consumption']):.2f} Joules"
            )
            print(
                " - Average power consumption: "
                f"{np.mean(feasibility['average_power']):.2f} "
                f"± {np.std(feasibility['average_power']):.2f} Watts"
            )

    # Cleanup energy monitor resources
    if "energy_monitor" in locals():
        energy_monitor.cleanup()


def _run_worker_refiner_pipeline(
    worker_model_name: str,
    refiner_model_name: str,
    dataset: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    max_model_len,
    worker_fine_tuned: bool,
    refiner_fine_tuned: bool,
    analyze_feasibility: bool,
    num_narratives: int,
):
    """Former `test_llm_refiner` pipeline (worker + refiner)."""

    print(
        f"[CONFIG] Worker+Refiner pipeline: "
        f"worker={worker_model_name} (fine_tuned={worker_fine_tuned}), "
        f"refiner={refiner_model_name} (fine_tuned={refiner_fine_tuned}), "
        f"dataset={dataset}, analyze_feasibility={analyze_feasibility}"
    )

    set_full_reproducibility()

    LOWER_BOUND = 1
    UPPER_BOUND = 200
    worker_name = worker_model_name
    refiner_name = refiner_model_name
    global prompt
    global prompt_ref
    base_prompt = prompt
    base_prompt_ref = prompt_ref
    mapped_worker_model_name = MODEL_MAPPING[worker_model_name]
    mapped_refiner_model_name = MODEL_MAPPING[refiner_model_name]

    # Decide if we can share model between worker and refiner (check once before loop)
    share_model = (
        worker_name == refiner_name
        and worker_fine_tuned == refiner_fine_tuned
    )

    # If sharing model, also share tokenizer
    if share_model:
        print("[SHARING] Using a single tokenizer instance for worker and refiner.")
        tokenizer_worker = AutoTokenizer.from_pretrained(mapped_worker_model_name)
        tokenizer_refiner = tokenizer_worker
    else:
        tokenizer_worker = AutoTokenizer.from_pretrained(mapped_worker_model_name)
        tokenizer_refiner = AutoTokenizer.from_pretrained(mapped_refiner_model_name)
    sampling_params_worker = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        repetition_penalty=repetition_penalty, 
        max_tokens=max_tokens, 
        top_k=10,
        stop=tokenizer_worker.eos_token
    )
    sampling_params_refiner = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        top_k=10,
        stop=tokenizer_refiner.eos_token
    )

    # Resolve LoRA checkpoints via CHECKPOINT_MAPPING
    worker_ckpt_step = get_checkpoint_step(dataset, "draft_generator", worker_name, default=500)
    refiner_ckpt_step = get_checkpoint_step(dataset, "refiner", refiner_name, default=800)

    lora_checkpoint_directory_path_worker = (
        f"outputs_unsloth/outputs_unsloth_{dataset}_worker/{worker_name}/checkpoint-{worker_ckpt_step}"
    )
    lora_checkpoint_directory_path_refiner = (
        f"outputs_unsloth/outputs_unsloth_{dataset}_refiner/{refiner_name}/checkpoint-{refiner_ckpt_step}"
    )

    print(
        f"[CHECKPOINT] Worker draft_generator checkpoint: "
        f"dataset={dataset}, model={worker_name}, step={worker_ckpt_step}"
    )
    print(
        f"[CHECKPOINT] Refiner checkpoint: "
        f"dataset={dataset}, model={refiner_name}, step={refiner_ckpt_step}"
    )

    # Calculate combined model metrics only once
    if analyze_feasibility:
        print("🔄 Calculating combined model metrics...")
        
        # Calculate worker model size using vLLM log capture
        log_capture_worker = VLLMLogCapture()
        handler_worker = log_capture_worker.capture_logs()
        
        print("🔄 Loading worker model...")
        worker_model_load_start = time.time()
        
        worker_llm_temp = LLM(
            model=mapped_worker_model_name, 
            gpu_memory_utilization=0.6,  
            max_model_len=max_model_len, 
            max_num_seqs=1,
            enable_lora=True
        )
        
        worker_model_load_end = time.time()
        worker_model_disk_size_gb = log_capture_worker.extract_model_size(handler_worker)
        
        # Get GPU memory after worker model loading
        gpu_memory_after_worker = get_gpu_memory_usage()

        del worker_llm_temp

        # Calculate refiner model size
        log_capture_refiner = VLLMLogCapture()
        handler_refiner = log_capture_refiner.capture_logs()
        
        print("🔄 Loading refiner model...")
        refiner_model_load_start = time.time()
        
        refiner_llm_temp = LLM(
            model=mapped_refiner_model_name,
            gpu_memory_utilization=0.6,  
            max_model_len=max_model_len, 
            max_num_seqs=1,
            enable_lora=refiner_fine_tuned,
        )
        
        refiner_model_load_end = time.time()
        refiner_model_disk_size_gb = log_capture_refiner.extract_model_size(handler_refiner)
        
        # Get GPU memory after both models loaded
        gpu_memory_after_both = get_gpu_memory_usage()
        
        # Calculate combined metrics
        combined_model_loading_time = (worker_model_load_end - worker_model_load_start) + (refiner_model_load_end - refiner_model_load_start)
        combined_model_disk_size_gb = worker_model_disk_size_gb + refiner_model_disk_size_gb
        combined_gpu_memory_usage_gb = gpu_memory_after_worker + gpu_memory_after_both  
        
        print(f"📊 Combined model loading time: {combined_model_loading_time:.2f} seconds")
        print(f"📊 Combined model weight size: {combined_model_disk_size_gb:.4f} GB (Worker: {worker_model_disk_size_gb:.4f} GB + Refiner: {refiner_model_disk_size_gb:.4f} GB)")
        print(f"📊 Combined GPU memory usage: {combined_gpu_memory_usage_gb:.2f} GB")
        
        # Clean up temporary model
        del refiner_llm_temp
        torch.cuda.empty_cache()

    # Define output file for results
    responses = {}  # Dictionary to store new responses
    
    # Initialize feasibility metrics if analyzing feasibility
    feasibility = None
    if analyze_feasibility:
        feasibility = {
            "loading_time": combined_model_loading_time,
            "model_weight_size_gb": combined_model_disk_size_gb,
            "gpu_memory_usage_gb": combined_gpu_memory_usage_gb,
            "worker_model_size_gb": worker_model_disk_size_gb,
            "refiner_model_size_gb": refiner_model_disk_size_gb,
            "inference_time": [],
            "energy_consumption": [],
            "average_power": [],
            "power_samples": [],
        }

    # Load counterfactual data
    with open(f"src/explainer/test_counterfactuals.json", 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    i = 0  # Counter for responses
    NUM_FACTUALS = 40  # Number of factuals to process
    NUM_COUNTERFACTUALS_PER_FACTUAL = 5  # Number of counterfactuals per factual
    if NUM_FACTUALS * NUM_COUNTERFACTUALS_PER_FACTUAL != UPPER_BOUND:
        raise ValueError(f"NUM_FACTUALS * NUM_COUNTERFACTUALS_PER_FACTUAL must be {UPPER_BOUND}")

    # Define output file base directory (with refiner)
    base_dir = os.path.join(
        "results",
        "with_refiner",
        dataset,
        f"{worker_name}--{refiner_name}",
    )
    os.makedirs(base_dir, exist_ok=True)

    # Define output files
    if not analyze_feasibility:
        output_file = os.path.join(
            base_dir,
            f"{worker_name}--{refiner_name}_response_finetuned_{worker_fine_tuned}-{refiner_fine_tuned}.json",
        )
        feasibility_output_file = None
    else:
        # When analyzing feasibility, save both test results and feasibility metrics
        output_file = os.path.join(
            base_dir,
            f"{worker_name}--{refiner_name}_response_finetuned_{worker_fine_tuned}-{refiner_fine_tuned}.json",
        )
        feasibility_output_file = os.path.join(
            base_dir,
            f"{worker_name}--{refiner_name}_feasibility_response_finetuned_{worker_fine_tuned}-{refiner_fine_tuned}.json",
        )

    print(f"[IO] Test results output file: {output_file}")
    if feasibility_output_file:
        print(f"[IO] Feasibility metrics output file: {feasibility_output_file}")

    # Initialize shared model ONCE before the loop if sharing is enabled
    shared_llm = None
    if share_model:
        print("[SHARING] Initializing a single LLM instance for worker and refiner (will be reused for all examples).")
        shared_llm = LLM(
            model=mapped_worker_model_name,
            gpu_memory_utilization=0.6,
            max_model_len=max_model_len,
            max_num_seqs=1,
            enable_lora=worker_fine_tuned or refiner_fine_tuned,
        )
        print("[SHARING] Shared model loaded successfully.")

    for dataset_name, examples in data1.items():

        dataset_key = dataset
        if dataset_key == "adult":
            dataset_key = "adult income"

        if dataset_name.lower() == dataset_key:
            print(f"[CONFIG] Processing {NUM_FACTUALS} factuals with {NUM_COUNTERFACTUALS_PER_FACTUAL} counterfactuals each (total: {NUM_FACTUALS * NUM_COUNTERFACTUALS_PER_FACTUAL} iterations)")

            # Sort indices to ensure consistent ordering (indices are strings like "0", "1", etc.)
            sorted_indices = sorted(examples.keys(), key=lambda x: int(x))[:NUM_FACTUALS]
            
            for index in sorted_indices:
                values = examples[index]
                # Process only the first NUM_COUNTERFACTUALS_PER_FACTUAL counterfactuals
                counterfactuals_to_process = values["counterfactuals"][:NUM_COUNTERFACTUALS_PER_FACTUAL]
                
                for counterfactual in counterfactuals_to_process:
                    
                    if LOWER_BOUND <= i <= UPPER_BOUND:
                        
                        current_prompt_worker = base_prompt.format(
                            dataset_description=dataset_kb[dataset_name], 
                            factual_example=str(values["factual"]), 
                            counterfactual_example=str(counterfactual)
                        )

                        messages = [{"role": "user", "content": current_prompt_worker}]
                        # Tokenize the messages for the worker LLM
                        text = tokenizer_worker.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        # Use shared model if available, otherwise create worker model for this example
                        if share_model:
                            worker_llm = shared_llm
                        else:
                            worker_llm = LLM(
                                model=mapped_worker_model_name,
                                gpu_memory_utilization=0.6,
                                max_model_len=max_model_len,
                                max_num_seqs=1,
                                enable_lora=worker_fine_tuned,
                            )
                        
                        # Initialize metrics tracking for this explanation
                        total_inference_time = 0.0
                        total_energy_consumed = 0.0
                        total_average_power = 0.0
                        total_power_samples = []
                        
                        # Generate explanations (draft narratives) using the worker LLM
                        explanations = []
                        for draft_num in range(num_narratives):
                            try:
                                # Initialize energy monitor for each draft
                                energy_monitor = EnergyMonitor() if analyze_feasibility else None
                                
                                with torch.no_grad():
                                    # Start energy monitoring and time measurement for draft
                                    if analyze_feasibility:
                                        energy_monitor.start_monitoring()
                                        draft_start = time.time()
                                    
                                    if worker_fine_tuned:
                                        outputs = worker_llm.generate(
                                            [text],
                                            sampling_params=sampling_params_worker,
                                            lora_request=LoRARequest(
                                                "counterfactual_explainability_adapter_worker",
                                                1,
                                                lora_checkpoint_directory_path_worker,
                                            ),
                                        )
                                    else:
                                        outputs = worker_llm.generate(
                                            [text],
                                            sampling_params=sampling_params_worker,
                                        )
                                    explanations.append(outputs)
                                    
                                    if analyze_feasibility:
                                        draft_end = time.time()
                                        energy_metrics = energy_monitor.stop_monitoring()
                                        
                                        # Accumulate metrics from this draft
                                        draft_inference_time = draft_end - draft_start
                                        draft_energy_consumed = energy_metrics['energy_joules']
                                        draft_average_power = energy_metrics['average_power_watts']
                                        draft_power_samples_tot = energy_metrics['power_samples']
                                        power_samples = []
                                        for j in range(len(draft_power_samples_tot)):
                                            power_samples.append(draft_power_samples_tot[j][1])

                                        
                                        total_inference_time += draft_inference_time
                                        total_energy_consumed += draft_energy_consumed
                                        total_average_power += draft_average_power
                                        total_power_samples.append(power_samples)

                                        print(f"Draft {draft_num + 1}/{num_narratives} - Time: {draft_inference_time:.2f}s, Energy: {draft_energy_consumed:.2f}J, Power: {draft_average_power:.2f}W")
                                        
                                        # Cleanup energy monitor
                                        energy_monitor.cleanup()
                                        
                            except AssertionError as assert_e:
                                print(f"🚨 Assertion error: {assert_e}")
                                continue
                        
                        # Build draft_narratives string for the refiner prompt
                        draft_narratives_parts = []
                        for idx, draft_outputs in enumerate(explanations):
                            # Each draft_outputs is a vLLM result list (same shape as in worker-only pipeline)
                            for out in draft_outputs:
                                draft_text = out.outputs[0].text
                            header = f"### Draft Explanation {idx + 1} ###"
                            draft_narratives_parts.append(f"{header}\n{draft_text}")
                        draft_narratives = "\n\n".join(draft_narratives_parts)

                        # Delete the worker LLM to free memory if not shared (only for non-shared case)
                        if not share_model:
                            del worker_llm
                            torch.cuda.empty_cache()

                        current_prompt_refiner = base_prompt_ref.format(
                            dataset_description=dataset_kb[dataset_name],
                            factual_example=str(values["factual"]),
                            counterfactual_example=str(counterfactual),
                            draft_narratives=draft_narratives,
                        )

                        messages = [{"role": "user", "content": current_prompt_refiner}]

                        text = tokenizer_refiner.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        # Use shared model if available, otherwise create refiner model for this example
                        if share_model:
                            refiner_llm = shared_llm
                        else:
                            refiner_llm = LLM(
                                model=mapped_refiner_model_name,
                                gpu_memory_utilization=0.6,
                                max_model_len=max_model_len,
                                max_num_seqs=1,
                                enable_lora=refiner_fine_tuned,
                            )

                        try:
                            # Initialize energy monitor for refiner
                            energy_monitor = EnergyMonitor() if analyze_feasibility else None
                            
                            with torch.no_grad():
                                # Start energy monitoring and time measurement for refiner
                                if analyze_feasibility:
                                    energy_monitor.start_monitoring()
                                    refiner_start = time.time()
                                
                                if refiner_fine_tuned:
                                    outputs = refiner_llm.generate(
                                        [text],
                                        sampling_params=sampling_params_refiner,
                                        lora_request=LoRARequest(
                                            "counterfactual_explainability_adapter_refiner",
                                            2,
                                            lora_checkpoint_directory_path_refiner,
                                        ),
                                    )
                                else:
                                    outputs = refiner_llm.generate(
                                        [text],
                                        sampling_params=sampling_params_refiner,
                                    )
                                
                                if analyze_feasibility:
                                    refiner_end = time.time()
                                    energy_metrics = energy_monitor.stop_monitoring()
                                    
                                    # Add refiner metrics to totals
                                    refiner_inference_time = refiner_end - refiner_start
                                    refiner_energy_consumed = energy_metrics['energy_joules']
                                    refiner_average_power = energy_metrics['average_power_watts']
                                    refiner_power_samples_tot = energy_metrics['power_samples']
                                    power_samples = []
                                    for j in range(len(refiner_power_samples_tot)):
                                        power_samples.append(refiner_power_samples_tot[j][1])

                                    
                                    total_inference_time += refiner_inference_time
                                    total_energy_consumed += refiner_energy_consumed
                                    total_average_power += refiner_average_power
                                    total_power_samples.append(power_samples)
                                    
                                    print(f"Refiner - Time: {refiner_inference_time:.2f}s, Energy: {refiner_energy_consumed:.2f}J, Power: {refiner_average_power:.2f}W")
                                    
                                    # Cleanup energy monitor
                                    energy_monitor.cleanup()
                        except AssertionError as assert_e:
                            print(f"🚨 Assertion error: {assert_e}")
                            continue

                        # Delete the refiner LLM to free memory if not shared (only for non-shared case)
                        if not share_model:
                            del refiner_llm
                            torch.cuda.empty_cache()

                        # Process the outputs if generated successfully
                        for output in outputs:
                            prompt = output.prompt
                            generated_text = output.outputs[0].text

                        print(generated_text)
                        
                        # Store response data
                        response_data = {
                            "generated_text": generated_text, 
                            "prompt": prompt, 
                            "ground_truth": {"counterfactual":counterfactual, "factual": values["factual"]}, 
                            "changes": extract_feature_changes(values["factual"], counterfactual)
                        }
                        responses[i] = response_data
                        
                        # Store metrics in feasibility dictionary if analyzing feasibility
                        if analyze_feasibility:
                            # Calculate average power across all 4 inferences (3 drafts + 1 refiner)
                            average_total_power = total_average_power / 4.0
                            
                            feasibility["inference_time"].append(total_inference_time)
                            feasibility["energy_consumption"].append(total_energy_consumed)
                            feasibility["average_power"].append(average_total_power)
                            all_power_samples = []
                            for samples in total_power_samples:
                                all_power_samples.extend(samples)
                            feasibility["power_samples"].append(all_power_samples)

                            print(f"#################### explanation #{i} - Combined Time taken: {total_inference_time:.2f}s, Total Energy: {total_energy_consumed:.2f}J, Avg Power: {average_total_power:.2f}W ###########################")
                        else:
                            print(f"#################### explanation #{i} completed - Time taken: {total_inference_time:.2f}s ###########################")

                        # Save every 10 responses (always save test results)
                        if i % 10 == 0:
                            save_responses(responses, output_file)
                            responses = {}  # Clear the temporary dictionary to prevent duplication

                        # Delete large variables to free memory
                        del generated_text
                        del outputs
                    i += 1
    
    # Cleanup shared model after all examples are processed
    if share_model and shared_llm is not None:
        print("[SHARING] Cleaning up shared model after processing all examples.")
        del shared_llm
        torch.cuda.empty_cache()
    
    # Final save after loop completion
    # Always save test results
    save_responses(responses, output_file)
    
    # If analyzing feasibility, also save feasibility metrics to separate file
    if analyze_feasibility:
        # Calculate statistics before saving
        if feasibility["inference_time"]:
            feasibility["inference_time_stats"] = {
                "mean": float(np.mean(feasibility["inference_time"])),
                "std": float(np.std(feasibility["inference_time"])),
                "min": float(np.min(feasibility["inference_time"])),
                "max": float(np.max(feasibility["inference_time"]))
            }
        
        if feasibility["energy_consumption"]:
            feasibility["energy_consumption_stats"] = {
                "mean": float(np.mean(feasibility["energy_consumption"])),
                "std": float(np.std(feasibility["energy_consumption"])),
                "min": float(np.min(feasibility["energy_consumption"])),
                "max": float(np.max(feasibility["energy_consumption"]))
            }
        
        if feasibility["average_power"]:
            feasibility["average_power_stats"] = {
                "mean": float(np.mean(feasibility["average_power"])),
                "std": float(np.std(feasibility["average_power"])),
                "min": float(np.min(feasibility["average_power"])),
                "max": float(np.max(feasibility["average_power"]))
            }
        
        # Save feasibility metrics to separate file
        with open(feasibility_output_file, 'w', encoding='utf-8') as f:
            json.dump(feasibility, f, indent=4)
        print(f"[FEASIBILITY] ✅ Feasibility metrics saved to {feasibility_output_file}")
        
        # Print summary statistics
        if feasibility["inference_time"]:
            print(f"\n📊 Feasibility Summary:")
            print(f"Combined model loading time: {combined_model_loading_time:.2f} seconds")
            print(f"Combined model weight size: {combined_model_disk_size_gb:.4f} GB")
            print(f"Combined GPU memory usage: {combined_gpu_memory_usage_gb:.2f} GB")
            print(f"Average combined inference time: {np.mean(feasibility['inference_time']):.2f} ± {np.std(feasibility['inference_time']):.2f} seconds")
            print(f"Average combined energy consumption: {np.mean(feasibility['energy_consumption']):.2f} ± {np.std(feasibility['energy_consumption']):.2f} Joules")
            print(f"Average combined power consumption: {np.mean(feasibility['average_power']):.2f} ± {np.std(feasibility['average_power']):.2f} Watts")
    
    # Cleanup energy monitor resources
    if 'energy_monitor' in locals() and analyze_feasibility:
        energy_monitor.cleanup()