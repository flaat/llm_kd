from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from data.dataset_kb import dataset_kb
import random
import numpy as np
from .utils import MODEL_MAPPING, prompt, prompt_ref
import json
import time
import os
import re
import ast
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
    """Appends new responses to the existing JSON file."""
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
                capture_output=True,
                text=True,
                timeout=2.0
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
            'energy_joules': total_energy,
            'average_power_watts': average_power,
            'duration_seconds': duration,
            'sample_count': len(self.power_samples)
        }

    def cleanup(self):
        """Cleanup resources (no-op for nvidia-smi version)."""
        pass


def _strip_reasoning_sections(text: str) -> str:
    # Remove reasoning blocks like <think>...</think>
    pattern = r"<(think|thinking|reasoning)[^>]*>.*?</\1>"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)


def _last_code_block_json(text: str):
    # Look for the last ```json ... ``` or generic ``` ... ```
    matches = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        matches = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def _last_balanced_json_object(text: str):
    # Find the last balanced JSON-like object
    last_obj = None
    stack = 0
    start_idx = None
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start_idx = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start_idx is not None:
                    candidate = text[start_idx:i + 1]
                    last_obj = candidate
                    start_idx = None
    return last_obj.strip() if last_obj else None


def _parse_json_loose(blob: str):
    # Try strict JSON, then a loose parse via ast.literal_eval
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(blob)
            if isinstance(obj, (dict, list)):
                return json.loads(json.dumps(obj))
        except Exception:
            return None
    except Exception:
        return None


def extract_single_narrative(outputs):
    """Extract a single narrative explanation from a generation result."""
    text = None
    for output in outputs:
        text = output.outputs[0].text
        break

    if not text:
        print("⚠️ Nessun testo generato nell'output.")
        return None

    cleaned = _strip_reasoning_sections(text)
    response = None

    blob = _last_code_block_json(cleaned)
    if blob:
        response = _parse_json_loose(blob)

    if response is None:
        blob = _last_balanced_json_object(cleaned)
        if blob:
            response = _parse_json_loose(blob)

    if isinstance(response, dict):
        if "explanation" in response and isinstance(response["explanation"], str):
            return response["explanation"]
        print(f"⚠️ Chiave 'explanation' non trovata. Chiavi disponibili: {list(response.keys())}")
        return None

    # Targeted regex fallback
    match = re.search(r'"explanation"\s*:\s*"(.+?)"', cleaned, flags=re.DOTALL)
    if match:
        return match.group(1).encode("utf-8").decode("unicode_escape")

    print(f"⚠️ Nessun JSON valido trovato. Estratto (inizio): {cleaned[:200]}...")
    return None


def build_dataset_wor(model_name: str, temperature: float, top_p: float, dataset: str, max_tokens: int,
                      repetition_penalty: float, max_model_len, analyze_feasibility: bool = True,
                      fine_tuned: bool = False, lora_checkpoint_path: str | None = None):
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
        max_num_seqs=1,
        enable_lora=fine_tuned,
    )

    # Define output files
    responses_output_file = f"data/{dataset}_Worker_{model_name}.json"
    feasibility_output_file = f"results/feasibility/{dataset}_worker_{model_name}.json"

    # Ensure results/feasibility directory exists
    os.makedirs("results/feasibility", exist_ok=True)

    responses = {}  # Dictionary to store new responses

    # Initialize feasibility metrics if analyzing feasibility
    if analyze_feasibility:
        feasibility = {
            "total_time_seconds": 0.0,
            "total_energy_joules": 0.0,
            "num_generations": 0,
            # For variance calculation (Welford's online algorithm)
            "time_M2": 0.0,
            "energy_M2": 0.0,
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
                                if fine_tuned:
                                    if not lora_checkpoint_path:
                                        raise ValueError("fine_tuned is True but no lora_checkpoint_path provided.")
                                    outputs = llm.generate(
                                        [text],
                                        sampling_params=sampling_params,
                                        lora_request=LoRARequest(
                                            "counterfactual_explainability_adapter_worker",
                                            1,
                                            lora_checkpoint_path,
                                        ),
                                    )
                                else:
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

                        # Save each response immediately
                        save_responses(responses, responses_output_file)
                        responses = {}

                        # Save feasibility stats periodically (every 100 generations)
                        if analyze_feasibility and i % 100 == 0 and feasibility["num_generations"] > 0:
                            avg_time = feasibility["total_time_seconds"] / feasibility["num_generations"]
                            avg_energy = feasibility["total_energy_joules"] / feasibility["num_generations"]
                            print(f"📊 Progress: {feasibility['num_generations']} generations - Avg time: {avg_time:.2f}s, Avg energy: {avg_energy:.2f}J")

                        # Delete large variables to free memory
                        del generated_text
                        del outputs
                    i += 1

    # Final save of responses after loop completion (no-op if empty)
    if responses:
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


def build_dataset_ref(worker_model_name: str, refiner_model_name: str, temperature: float, top_p: float,
                      dataset: str, max_tokens: int, repetition_penalty: float, max_model_len,
                      number_narratives: int, fine_tuned: bool = False, lora_checkpoint_path: str | None = None,
                      analyze_feasibility: bool = True):
    set_full_reproducibility()

    LOWER_BOUND = 0
    UPPER_BOUND = 50000

    global prompt
    global prompt_ref
    base_prompt = prompt
    base_prompt_ref = prompt_ref

    # Map model names to HuggingFace identifiers
    worker_model_path = MODEL_MAPPING[worker_model_name]
    refiner_model_path = MODEL_MAPPING[refiner_model_name]

    # Initialize tokenizers for both models
    worker_tokenizer = AutoTokenizer.from_pretrained(worker_model_path)
    refiner_tokenizer = AutoTokenizer.from_pretrained(refiner_model_path)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        top_k=10
    )

    print(f"📋 Worker Model: {worker_model_name}")
    print(f"📋 Refiner Model: {refiner_model_name}")
    print(f"🔄 Models will be loaded sequentially to avoid GPU memory issues")

    # Define output file for results (keep original location, but use worker--refiner naming)
    output_file = f"data/{dataset}_Refiner_{worker_model_name}--{refiner_model_name}.json"
    responses = {}  # Dictionary to store new responses

    feasibility = None
    feasibility_output_file = None
    if analyze_feasibility:
        os.makedirs("results/feasibility", exist_ok=True)
        feasibility_output_file = os.path.join(
            "results",
            "feasibility",
            f"{dataset}_Refiner_{worker_model_name}--{refiner_model_name}.json",
        )
        feasibility = {
            "total_time_seconds": 0.0,
            "total_energy_joules": 0.0,
            "num_generations": 0,
            "time_M2": 0.0,
            "energy_M2": 0.0,
            "time_mean": 0.0,
            "energy_mean": 0.0,
        }

    # Load counterfactual data
    with open(f"src/explainer/counterfactuals.json", 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    i = 0  # Counter for responses

    if dataset == "adult":
        dataset = "adult income"
    elif dataset == "california":
        dataset = "california housing"

    for dataset_name, examples in data1.items():

        if dataset_name.lower() == dataset:
            for index, values in examples.items():
                for counterfactual in values["counterfactuals"]:

                    if LOWER_BOUND <= i <= UPPER_BOUND:

                        current_prompt_worker = base_prompt.format(
                            dataset_description=dataset_kb[dataset_name],
                            factual_example=str(values["factual"]),
                            counterfactual_example=str(counterfactual)
                        )

                        messages = [{"role": "user", "content": current_prompt_worker}]

                        text_worker = worker_tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )

                        # Start timing for the entire process (draft + refinement)
                        total_start_time = time.time()
                        total_energy_consumed = 0.0

                        # Load worker model
                        print(f"🔧 Loading Worker Model: {worker_model_name}")
                        worker_llm = LLM(
                            model=worker_model_path,
                            gpu_memory_utilization=0.5,
                            max_model_len=max_model_len,
                            max_num_seqs=1,
                            enable_lora=fine_tuned,
                        )

                        # Generate and parse draft narratives iteratively with one retry per narrative
                        N = number_narratives
                        narratives = []
                        for j in range(N):
                            attempts = 0
                            narrative = None
                            while attempts < 2 and narrative is None:
                                try:
                                    with torch.no_grad():
                                        print(f"🔄 Generating draft narrative {j + 1} attempt {attempts + 1} of 2 using worker model")
                                        draft_start = time.time()
                                        if fine_tuned:
                                            if not lora_checkpoint_path:
                                                raise ValueError("fine_tuned is True but no lora_checkpoint_path provided.")
                                            if analyze_feasibility:
                                                energy_monitor = EnergyMonitor()
                                                energy_monitor.start_monitoring()
                                            outputs = worker_llm.generate(
                                                [text_worker],
                                                sampling_params=sampling_params,
                                                lora_request=LoRARequest(
                                                    "counterfactual_explainability_adapter_worker",
                                                    1,
                                                    lora_checkpoint_path,
                                                ),
                                            )
                                            if analyze_feasibility:
                                                energy_metrics = energy_monitor.stop_monitoring()
                                                total_energy_consumed += energy_metrics["energy_joules"]
                                        else:
                                            if analyze_feasibility:
                                                energy_monitor = EnergyMonitor()
                                                energy_monitor.start_monitoring()
                                            outputs = worker_llm.generate([text_worker], sampling_params=sampling_params)
                                            if analyze_feasibility:
                                                energy_metrics = energy_monitor.stop_monitoring()
                                                total_energy_consumed += energy_metrics["energy_joules"]
                                        for output in outputs:
                                            prompt = output.prompt
                                            generated_text = output.outputs[0].text
                                        print(generated_text)
                                        narrative = extract_single_narrative(outputs)
                                        draft_end = time.time()
                                        print(f"⏱️  Draft narrative {j + 1} attempt {attempts + 1} completed in {draft_end - draft_start:.2f}s")
                                except AssertionError as assert_e:
                                    print(f"🚨 Assertion error: {assert_e}")
                                    break
                                attempts += 1
                            narratives.append(narrative)

                        # Delete the worker LLM to free GPU memory
                        if worker_model_name != refiner_model_name:
                            print(f"🗑️  Unloading worker model to free GPU memory")
                            del worker_llm
                            torch.cuda.empty_cache()

                        # Dynamically construct draft narratives section
                        draft_narratives_text = ""
                        for idx, narrative in enumerate(narratives, start=1):
                            if narrative is not None:
                                draft_narratives_text += f"### Draft Narrative {idx} ###\n{narrative}\n"
                            else:
                                draft_narratives_text += f"### Draft Narrative {idx} ###\nNone\n"

                        current_prompt_refiner = base_prompt_ref.format(
                            dataset_description=dataset_kb[dataset_name],
                            factual_example=str(values["factual"]),
                            counterfactual_example=str(counterfactual),
                            draft_narratives=draft_narratives_text.strip()
                        )

                        messages = [{"role": "user", "content": current_prompt_refiner}]

                        # Always apply chat template for refiner with draft narratives
                        if worker_model_name != refiner_model_name:
                            text_refiner = refiner_tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                        else:
                            # Use worker tokenizer for same model, but with refiner prompt
                            text_refiner = worker_tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )

                        # Load refiner model
                        print(f"🔧 Loading Refiner Model: {refiner_model_name}")
                        if worker_model_name != refiner_model_name:
                            refiner_llm = LLM(
                                model=refiner_model_path,
                                gpu_memory_utilization=0.92,
                                max_model_len=max_model_len,
                                max_num_seqs=1
                            )
                        else:
                            refiner_llm = worker_llm

                        try:
                            with torch.no_grad():
                                print(f"✨ Refining narratives using refiner model")
                                refine_start = time.time()
                                if analyze_feasibility:
                                    energy_monitor = EnergyMonitor()
                                    energy_monitor.start_monitoring()
                                outputs = refiner_llm.generate([text_refiner], sampling_params=sampling_params)
                                if analyze_feasibility:
                                    energy_metrics = energy_monitor.stop_monitoring()
                                    total_energy_consumed += energy_metrics["energy_joules"]
                                refine_end = time.time()
                                print(f"⏱️  Refinement completed in {refine_end - refine_start:.2f}s")
                        except AssertionError as assert_e:
                            print(f"🚨 Assertion error: {assert_e}")
                            continue

                        # Delete the refiner LLM to free GPU memory
                        print(f"🗑️  Unloading refiner model to free GPU memory")
                        del refiner_llm
                        torch.cuda.empty_cache()

                        # Calculate total time for draft generation + refinement
                        total_end_time = time.time()
                        total_time = total_end_time - total_start_time

                        # Process the outputs if generated successfully
                        for output in outputs:
                            prompt = output.prompt
                            generated_text = output.outputs[0].text

                        print(generated_text)
                        responses[i] = {"prompt": prompt, "generated_text": generated_text}

                        print(f"#################### Total time taken: {total_time:.2f} seconds (draft + refinement), explanation number {i} ###########################")

                        # Update feasibility metrics if enabled
                        if analyze_feasibility and feasibility is not None:
                            feasibility["num_generations"] += 1
                            n = feasibility["num_generations"]

                            feasibility["total_time_seconds"] += total_time
                            feasibility["total_energy_joules"] += total_energy_consumed

                            delta_time = total_time - feasibility["time_mean"]
                            feasibility["time_mean"] += delta_time / n
                            delta_time2 = total_time - feasibility["time_mean"]
                            feasibility["time_M2"] += delta_time * delta_time2

                            delta_energy = total_energy_consumed - feasibility["energy_mean"]
                            feasibility["energy_mean"] += delta_energy / n
                            delta_energy2 = total_energy_consumed - feasibility["energy_mean"]
                            feasibility["energy_M2"] += delta_energy * delta_energy2

                        # Save each response immediately
                        save_responses(responses, output_file)
                        responses = {}  # Clear the temporary dictionary to prevent duplication

                        # Delete large variables to free memory
                        del generated_text
                        del outputs
                    i += 1
    # Final save after loop completion (no-op if empty)
    if responses:
        save_responses(responses, output_file)

    # Save feasibility statistics if enabled
    if analyze_feasibility and feasibility is not None and feasibility["num_generations"] > 0:
        n = feasibility["num_generations"]

        time_variance = feasibility["time_M2"] / n if n > 1 else 0.0
        energy_variance = feasibility["energy_M2"] / n if n > 1 else 0.0
        time_std = np.sqrt(time_variance)
        energy_std = np.sqrt(energy_variance)

        stats = {
            "total_time_seconds": feasibility["total_time_seconds"],
            "total_energy_joules": feasibility["total_energy_joules"],
            "average_inference_time_seconds": feasibility["total_time_seconds"] / n,
            "average_energy_per_generation_joules": feasibility["total_energy_joules"] / n,
            "std_inference_time_seconds": time_std,
            "std_energy_per_generation_joules": energy_std,
            "num_generations": n,
        }

        if feasibility_output_file:
            with open(feasibility_output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=4)
            print(f"✅ Feasibility statistics saved to {feasibility_output_file}")

        print(f"\n📊 Feasibility Summary:")
        print(f"Total time: {stats['total_time_seconds']:.2f} seconds")
        print(f"Total energy: {stats['total_energy_joules']:.2f} Joules")
        print(f"Average inference time per generation: {stats['average_inference_time_seconds']:.2f} ± {stats['std_inference_time_seconds']:.2f} seconds")
        print(f"Average energy per generation: {stats['average_energy_per_generation_joules']:.2f} ± {stats['std_energy_per_generation_joules']:.2f} Joules")
        print(f"Number of generations: {stats['num_generations']}")

