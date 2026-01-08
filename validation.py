import unsloth
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
import gc
from data.dataset_kb import dataset_kb
from src.utils import MODEL_MAPPING, prompt, prompt_ref, get_checkpoint_step
from src.build_dataset import extract_single_narrative
import re
from vllm.lora.request import LoRARequest
import argparse
from unsloth.chat_templates import get_chat_template

def set_full_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def display_config(args: argparse.Namespace) -> None:
    """
    Display the experiment configuration parameters.
    
    Args:
        args: Parsed command-line arguments
    """
    print("\n========== EXPERIMENT CONFIGURATION ==========")
    print(f"Worker model name:   {args.worker_model_name}")
    print(f"Refiner model name:  {args.refiner_model_name}")
    print(f"Model context length: {args.max_model_len}")
    print(f"Using fine-tuned:    {'Yes' if args.fine_tuned else 'No'}")
    print(f"Using refiner:      {'Yes' if args.refiner else 'No'}")
    print(f"Checkpoint every:     {args.checkpoint_every}")
    print(f"Max checkpoint:       {args.max_checkpoint}")
    print("\n----- Generation Parameters -----")
    print(f"Temperature:         {args.temperature}")
    print(f"Top-p:               {args.top_p}")
    print(f"Max tokens:          {args.max_tokens}")
    print(f"Repetition penalty:  {args.repetition_penalty}")
    if args.refiner:
        print(f"Number of narratives: {args.number_narratives}")
    print("\n----- Experiment Settings -----")
    print(f"Dataset:             {args.dataset}")
    if hasattr(args, 'num_counterfactuals_per_factual'):
        print(f"Counterfactuals per factual: {args.num_counterfactuals_per_factual}")
    print("=============================================\n")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the experiment runner.
    
    Returns:
        Namespace containing all parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run language model experiments with configurable parameters."
    )

    # Model configuration parameters
    parser.add_argument(
        '--worker_model_name',
        type=str,
        default='phi_4B',
        help='Worker model name to use for experiments (default: phi_4B)'
    )

    parser.add_argument(
        '--refiner_model_name',
        type=str,
        default='phi_4B',
        help='Refiner model name to use for experiments (default: phi_4B)'
    )

    parser.add_argument(
        '--max_model_len',
        type=int,
        default=8192,
        help='Maximum context length for the model (default: 8192)'
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=5000,
        help='Maximum number of tokens to generate (default: 5500)'
    )
    
    parser.add_argument(
        '--fine_tuned',
        action='store_true',
        help='Use fine-tuned version of the model instead of base model'
    )

    parser.add_argument(
        '--refiner',
        action='store_true',
        help='Use refiner model for generating explanations'
    )

    # Generation parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Temperature for text generation - higher values increase randomness (default: 0.6)'
    )
    
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help='Top-p (nucleus) sampling parameter - lower values increase determinism (default: 0.8)'
    )
    
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.05,
        help='Penalty applied to repeating tokens - higher values discourage repetition (default: 1.05)'
    )

    # Data and experiment type parameters
    parser.add_argument(
        '--dataset',
        type=str,
        default='adult',
        choices=['adult', 'california', 'titanic', 'diabetes'],
        help='Dataset to use for experiments (default: adult)'
    )

    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=20,
        help='Number of iterations between saving checkpoints (default: 50)'
    )

    parser.add_argument(
        '--max_checkpoint',
        type=int,
        default=None,
        help='Maximum checkpoint number (auto-detected if not provided)'
    )

    parser.add_argument(
        '--number_narratives',
        type=int,
        default=5,
        help='Number of draft narratives to generate for refiner (default: 5)'
    )

    parser.add_argument(
        '--num_counterfactuals_per_factual',
        type=int,
        default=2,
        help='Number of counterfactuals to process per factual (default: 2)'
    )

    return parser.parse_args()

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

def extract_max_checkpoint(output_dir: str) -> int:
    """
    Extracts the maximum checkpoint number from the output directory.
    
    Args:
        output_dir: Path to the output directory containing checkpoint folders
        
    Returns:
        The maximum checkpoint number found, or raises an error if none exists
    """
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {output_dir}")
    
    # Extract numbers from checkpoint directory names and find the maximum
    checkpoint_numbers = []
    for checkpoint_dir in checkpoint_dirs:
        try:
            num = int(checkpoint_dir.split('-')[1])
            checkpoint_numbers.append(num)
        except (ValueError, IndexError):
            continue
    
    if not checkpoint_numbers:
        raise ValueError(f"No valid checkpoint numbers found in {output_dir}")
    
    return max(checkpoint_numbers)


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



def validate_worker(model_name: str, dataset: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float, max_model_len, fine_tuned=True, checkpoint_every=50, max_checkpoint=None, num_counterfactuals_per_factual=2):

    set_full_reproducibility()
    
    # Auto-detect max_checkpoint if not provided
    if max_checkpoint is None:
        output_base_dir = f"outputs_unsloth/outputs_unsloth_{dataset}_worker/{model_name}"
        max_checkpoint = extract_max_checkpoint(output_base_dir)
        print(f"Auto-detected max_checkpoint: {max_checkpoint}")

    print(f"Params list: {model_name}, {temperature}, {top_p}, {max_tokens}, {repetition_penalty}, {max_model_len}, {fine_tuned}, {checkpoint_every}, {max_checkpoint}")

    # Check if the checkpoint directory exists
    lora_max_checkpoint_directory_path = f"outputs_unsloth/outputs_unsloth_{dataset}_worker/{model_name}/checkpoint-{max_checkpoint}"
    if not os.path.exists(lora_max_checkpoint_directory_path):
        raise ValueError(f"Max checkpoint value wrong: {lora_max_checkpoint_directory_path} does not exist")
    
    LOWER_BOUND = 0
    UPPER_BOUND = 19
    NUM_FACTUALS = (UPPER_BOUND + 1) // num_counterfactuals_per_factual
    if NUM_FACTUALS * num_counterfactuals_per_factual <= UPPER_BOUND:
        NUM_FACTUALS += 1
    
    name = model_name
    global prompt
    base_prompt = prompt
    model_name = MODEL_MAPPING[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        repetition_penalty=repetition_penalty, 
        max_tokens=max_tokens, 
        top_k=10,
        stop=tokenizer.eos_token
    )

    # Initialize LLM with optimized GPU memory usage
    if fine_tuned:
        llm = LLM(
        model=model_name, 
        gpu_memory_utilization=0.5, 
        max_model_len=max_model_len, 
        max_num_seqs=1,
        enable_lora=True,
    )
    else:
        llm = LLM(
            model=model_name, 
            gpu_memory_utilization=0.7, 
            max_model_len=max_model_len, 
            max_num_seqs=1
        )

    checkpoint_steps = list(range(checkpoint_every, max_checkpoint + 1, checkpoint_every))
    if checkpoint_steps[-1] != max_checkpoint:
        checkpoint_steps.append(max_checkpoint)

    old_dataset_name = dataset
    if dataset == "adult":
        dataset = "adult income"
    if dataset == "california":
        dataset = "california housing"
    
    for checkpoint in checkpoint_steps:

        print(f"🔄 Loading checkpoint {checkpoint}...")
        lora_checkpoint_directory_path = f"outputs_unsloth/outputs_unsloth_{old_dataset_name}_worker/{name}/checkpoint-{checkpoint}"

        # Define output file for results
        responses = {}  # Dictionary to store new responses
        # Load counterfactual data
        with open(f"src/explainer/val_counterfactuals.json", 'r', encoding='utf-8') as file1:
            data = json.load(file1)

        i = 0  # Counter for responses

        for dataset_name, examples in data.items():
            output_directory = f"results/fine-tuning/worker_validation/{old_dataset_name}/{name}"
            os.makedirs(output_directory, exist_ok=True)
            output_file = f"{output_directory}/{name}_checkpoint_{checkpoint}.json"
            
            if dataset_name.lower() == dataset:
                print(f"[CONFIG] Processing {NUM_FACTUALS} factuals with {num_counterfactuals_per_factual} counterfactuals each")
                
                # Process first NUM_FACTUALS factuals (data is already in order)
                indices = list(examples.keys())[:NUM_FACTUALS]
                
                for index in indices:
                    values = examples[index]
                    # Process only the first num_counterfactuals_per_factual counterfactuals
                    counterfactuals_to_process = values["counterfactuals"][:num_counterfactuals_per_factual]
                    
                    for counterfactual in counterfactuals_to_process:
                        
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
                            try:
                                
                                with torch.no_grad():
                                    start_time = time.time()
                                    if fine_tuned:
                                        adapter_name = f"counterfactual_explainability_adapter_{checkpoint}"
                                        adapter_id = checkpoint  # unique id so vLLM does not reuse a cached adapter
                                        outputs = llm.generate(
                                            [text],
                                            sampling_params=sampling_params,
                                            lora_request=LoRARequest(adapter_name, adapter_id, lora_checkpoint_directory_path),
                                        )
                                    else:
                                        outputs = llm.generate([text], sampling_params=sampling_params)
                                    end_time = time.time()
                                    generation_time = end_time - start_time
                                    
                                    
                            except AssertionError as assert_e:
                                print(f"🚨 Assertion error: {assert_e}")
                                continue
                            

                            # Process the outputs if generated successfully
                            for output in outputs:
                                prompt = output.prompt
                                generated_text = output.outputs[0].text

                            print(generated_text)
                            
                            # Store response with metrics
                            response_data = {
                                "generated_text": generated_text, 
                                "prompt": prompt, 
                                "ground_truth": {"counterfactual":counterfactual, "factual": values["factual"]}, 
                                "changes": extract_feature_changes(values["factual"], counterfactual),
                            }
                            responses[i] = response_data

                            print(f"#################### explanation #{i} (Generation time: {generation_time:.2f} seconds) ###########################")

                            # Save every 10 responses
                            if i % 10 == 0:
                                save_responses(responses, output_file)
                                responses = {}  # Clear the temporary dictionary to prevent duplication

                            # Delete large variables to free memory
                            del generated_text
                            del outputs
                            
                        i += 1

        save_responses(responses, output_file)
        #del llm
        #torch.cuda.empty_cache()


def validate_refiner(worker_model_name: str, refiner_model_name: str, dataset: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float, max_model_len: int, number_narratives: int, num_counterfactuals_per_factual: int, fine_tuned: bool, checkpoint_every: int, max_checkpoint: int):
    
    set_full_reproducibility()
    
    # Auto-detect max_checkpoint for refiner if not provided
    if max_checkpoint is None:
        output_base_dir = f"outputs_unsloth_{dataset}_refiner/{refiner_model_name}"
        max_checkpoint = extract_max_checkpoint(output_base_dir)
        print(f"Auto-detected max_checkpoint for refiner: {max_checkpoint}")
    
    # Get fixed worker checkpoint using get_checkpoint_step (same as pipeline.py)
    worker_checkpoint_step = get_checkpoint_step(dataset, "draft_generator", worker_model_name, default=500)
    print(f"Using fixed worker checkpoint: {worker_checkpoint_step}")
    
    print(f"Params list: worker={worker_model_name}, refiner={refiner_model_name}, {temperature}, {top_p}, {max_tokens}, {repetition_penalty}, {max_model_len}, {fine_tuned}, {checkpoint_every}, {max_checkpoint}, narratives={number_narratives}")
    
    # Check if checkpoints exist
    worker_checkpoint_directory_path = f"outputs_unsloth/outputs_unsloth_{dataset}_worker/{worker_model_name}/checkpoint-{worker_checkpoint_step}"
    if not os.path.exists(worker_checkpoint_directory_path):
        raise ValueError(f"Worker checkpoint does not exist: {worker_checkpoint_directory_path}")
    
    LOWER_BOUND = 0
    UPPER_BOUND = 19
    NUM_FACTUALS = (UPPER_BOUND + 1) // num_counterfactuals_per_factual
    if NUM_FACTUALS * num_counterfactuals_per_factual <= UPPER_BOUND:
        NUM_FACTUALS += 1
    
    worker_name = worker_model_name
    refiner_name = refiner_model_name
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
    
    sampling_params_worker = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        top_k=10,
        stop=worker_tokenizer.eos_token
    )
    
    sampling_params_refiner = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        top_k=10,
        stop=refiner_tokenizer.eos_token
    )
    
    checkpoint_steps = list(range(checkpoint_every, max_checkpoint + 1, checkpoint_every))
    if checkpoint_steps[-1] != max_checkpoint:
        checkpoint_steps.append(max_checkpoint)
    
    old_dataset_name = dataset
    if dataset == "adult":
        dataset = "adult income"
    elif dataset == "california":
        dataset = "california housing"
    
    for checkpoint in checkpoint_steps:
        
        print(f"🔄 Loading refiner checkpoint {checkpoint}...")
        refiner_checkpoint_directory_path = f"outputs_unsloth_{old_dataset_name}_refiner/{refiner_name}/checkpoint-{checkpoint}"
        
        if not os.path.exists(refiner_checkpoint_directory_path):
            print(f"⚠️ Warning: Refiner checkpoint {checkpoint} does not exist, skipping...")
            continue
        
        # Load worker model once per checkpoint (before processing samples)
        print(f"🔧 Loading Worker Model: {worker_model_name} (checkpoint {worker_checkpoint_step})")
        # Use lower GPU memory utilization to account for residual memory from previous checkpoints
        worker_llm = LLM(
            model=worker_model_path,
            gpu_memory_utilization=0.5,
            max_model_len=max_model_len,
            max_num_seqs=1,
            enable_lora=fine_tuned,
        )
        
        # Load refiner model once per checkpoint (before processing samples)
        print(f"🔧 Loading Refiner Model: {refiner_model_name} (checkpoint {checkpoint})")
        if worker_model_name != refiner_model_name:
            refiner_llm = LLM(
                model=refiner_model_path,
                gpu_memory_utilization=0.5,
                max_model_len=max_model_len,
                max_num_seqs=1,
                enable_lora=fine_tuned,
            )
        else:
            refiner_llm = worker_llm
        
        # Define output file for results
        responses = {}  # Dictionary to store new responses
        # Load counterfactual data
        with open(f"src/explainer/val_counterfactuals.json", 'r', encoding='utf-8') as file1:
            data = json.load(file1)
        
        i = 0  # Counter for responses
        
        for dataset_name, examples in data.items():
            output_directory = f"results/fine-tuning/refiner_validation/{old_dataset_name}/{worker_name}--{refiner_name}"
            os.makedirs(output_directory, exist_ok=True)
            output_file = f"{output_directory}/{worker_name}--{refiner_name}_checkpoint_{checkpoint}.json"
            
            if dataset_name.lower() == dataset:
                print(f"[CONFIG] Processing {NUM_FACTUALS} factuals with {num_counterfactuals_per_factual} counterfactuals each")
                
                # Process first NUM_FACTUALS factuals (data is already in order)
                indices = list(examples.keys())[:NUM_FACTUALS]
                
                for index in indices:
                    values = examples[index]
                    # Process only the first num_counterfactuals_per_factual counterfactuals
                    counterfactuals_to_process = values["counterfactuals"][:num_counterfactuals_per_factual]
                    
                    for counterfactual in counterfactuals_to_process:
                        
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
                            
                            # Generate and parse draft narratives iteratively
                            N = number_narratives
                            narratives = []
                            print(f"\n📝 [DRAFT GENERATION] Starting generation of {N} draft narratives for sample #{i}")
                            print(f"   Worker model: {worker_model_name}, Checkpoint: {worker_checkpoint_step}")
                            
                            for j in range(N):
                                attempts = 0
                                narrative = None
                                while attempts < 2 and narrative is None:
                                    try:
                                        with torch.no_grad():
                                            print(f"🔄 [DRAFT {j + 1}/{N}] Generating draft narrative {j + 1}, attempt {attempts + 1} of 2 using worker model")
                                            start_time = time.time()
                                            if fine_tuned:
                                                adapter_name = f"counterfactual_explainability_adapter_worker_{worker_checkpoint_step}"
                                                adapter_id = worker_checkpoint_step
                                                outputs = worker_llm.generate(
                                                    [text_worker],
                                                    sampling_params=sampling_params_worker,
                                                    lora_request=LoRARequest(
                                                        adapter_name,
                                                        adapter_id,
                                                        worker_checkpoint_directory_path,
                                                    ),
                                                )
                                            else:
                                                outputs = worker_llm.generate([text_worker], sampling_params=sampling_params_worker)
                                            end_time = time.time()
                                            
                                            for output in outputs:
                                                generated_text = output.outputs[0].text
                                            print(f"   [DRAFT {j + 1}/{N}] Raw output (first 200 chars): {generated_text[:200]}...")
                                            narrative = extract_single_narrative(outputs)
                                            
                                            if narrative:
                                                print(f"✅ [DRAFT {j + 1}/{N}] Successfully extracted narrative (length: {len(narrative)} chars)")
                                                print(f"   [DRAFT {j + 1}/{N}] Narrative preview: {narrative[:150]}...")
                                            else:
                                                print(f"⚠️  [DRAFT {j + 1}/{N}] Failed to extract narrative from output")
                                            
                                            print(f"⏱️  [DRAFT {j + 1}/{N}] Generation time: {end_time - start_time:.2f}s")
                                    except AssertionError as assert_e:
                                        print(f"🚨 [DRAFT {j + 1}/{N}] Assertion error: {assert_e}")
                                        break
                                    attempts += 1
                                narratives.append(narrative)
                            
                            # Log summary of draft narratives
                            successful_drafts = sum(1 for n in narratives if n is not None)
                            print(f"\n📊 [DRAFT GENERATION SUMMARY] Successfully generated {successful_drafts}/{N} draft narratives")
                            for idx, narrative in enumerate(narratives, start=1):
                                status = "✅ Success" if narrative else "❌ Failed"
                                length = len(narrative) if narrative else 0
                                print(f"   Draft {idx}: {status} (length: {length} chars)")
                            
                            # Dynamically construct draft narratives section
                            draft_narratives_text = ""
                            for idx, narrative in enumerate(narratives, start=1):
                                if narrative is not None:
                                    draft_narratives_text += f"### Draft Narrative {idx} ###\n{narrative}\n"
                                else:
                                    draft_narratives_text += f"### Draft Narrative {idx} ###\nNone\n"
                            
                            print(f"\n📋 [REFINEMENT PREP] Constructed draft narratives text (length: {len(draft_narratives_text)} chars)")
                            
                            current_prompt_refiner = base_prompt_ref.format(
                                dataset_description=dataset_kb[dataset_name],
                                factual_example=str(values["factual"]),
                                counterfactual_example=str(counterfactual),
                                draft_narratives=draft_narratives_text.strip()
                            )
                            
                            messages = [{"role": "user", "content": current_prompt_refiner}]
                            
                            # Apply chat template for refiner with draft narratives
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
                            
                            print(f"📝 [REFINEMENT] Starting refinement process for sample #{i}")
                            print(f"   Refiner model: {refiner_model_name}, Checkpoint: {checkpoint}")
                            print(f"   Prompt length: {len(text_refiner)} chars")
                            
                            try:
                                with torch.no_grad():
                                    print(f"✨ [REFINEMENT] Generating refined narrative using refiner model")
                                    start_time = time.time()
                                    if fine_tuned:
                                        adapter_name = f"counterfactual_explainability_adapter_refiner_{checkpoint}"
                                        adapter_id = checkpoint
                                        outputs = refiner_llm.generate(
                                            [text_refiner],
                                            sampling_params=sampling_params_refiner,
                                            lora_request=LoRARequest(
                                                adapter_name,
                                                adapter_id,
                                                refiner_checkpoint_directory_path,
                                            ),
                                        )
                                    else:
                                        outputs = refiner_llm.generate([text_refiner], sampling_params=sampling_params_refiner)
                                    end_time = time.time()
                                    generation_time = end_time - start_time
                                    
                                    # Process the outputs if generated successfully
                                    for output in outputs:
                                        prompt = output.prompt
                                        generated_text = output.outputs[0].text
                                    
                                    print(f"✅ [REFINEMENT] Refinement completed in {generation_time:.2f}s")
                                    print(f"   [REFINEMENT] Refined output length: {len(generated_text)} chars")
                                    print(f"   [REFINEMENT] Refined output preview: {generated_text[:200]}...")
                                    
                                    print(f"\n📊 [SAMPLE #{i} SUMMARY]")
                                    print(f"   Draft narratives: {successful_drafts}/{N} successful")
                                    print(f"   Refinement time: {generation_time:.2f}s")
                                    print(f"   Final output length: {len(generated_text)} chars")
                                    print(f"   Full refined output:\n{generated_text}\n")
                            except AssertionError as assert_e:
                                print(f"🚨 [REFINEMENT] Assertion error: {assert_e}")
                                continue
                            
                            # Store response with metrics
                            response_data = {
                                "generated_text": generated_text,
                                "prompt": prompt,
                                "ground_truth": {"counterfactual": counterfactual, "factual": values["factual"]},
                                "changes": extract_feature_changes(values["factual"], counterfactual),
                            }
                            responses[i] = response_data
                            
                            print(f"#################### explanation #{i} (Generation time: {generation_time:.2f} seconds) ###########################")
                            
                            # Save every 10 responses
                            if i % 10 == 0:
                                save_responses(responses, output_file)
                                responses = {}  # Clear the temporary dictionary to prevent duplication
                            
                            # Delete large variables to free memory
                            del generated_text
                            del outputs
                            
                        i += 1
        
        save_responses(responses, output_file)
        
        # Cleanup models after processing all samples for this checkpoint
        # This frees GPU memory before loading the next checkpoint
        print(f"🗑️  [CLEANUP] Unloading models after processing checkpoint {checkpoint}")
        
        # Properly shutdown vLLM engines to release GPU memory and background processes
        if worker_model_name != refiner_model_name:
            # Shutdown refiner_llm if it's a separate model
            try:
                # Try to shutdown the engine if it exists
                if hasattr(refiner_llm, 'llm_engine') and refiner_llm.llm_engine is not None:
                    if hasattr(refiner_llm.llm_engine, 'shutdown'):
                        refiner_llm.llm_engine.shutdown()
                    elif hasattr(refiner_llm.llm_engine, 'engine_core') and refiner_llm.llm_engine.engine_core is not None:
                        # For vLLM v1 engine
                        if hasattr(refiner_llm.llm_engine.engine_core, 'shutdown'):
                            refiner_llm.llm_engine.engine_core.shutdown()
            except Exception as e:
                print(f"⚠️  [CLEANUP] Warning during refiner model shutdown: {e}")
            finally:
                del refiner_llm
        
        # Always shutdown worker_llm (it's always loaded)
        # If worker and refiner are the same model, this will shutdown the shared object
        try:
            # Try to shutdown the engine if it exists
            if hasattr(worker_llm, 'llm_engine') and worker_llm.llm_engine is not None:
                if hasattr(worker_llm.llm_engine, 'shutdown'):
                    worker_llm.llm_engine.shutdown()
                elif hasattr(worker_llm.llm_engine, 'engine_core') and worker_llm.llm_engine.engine_core is not None:
                    # For vLLM v1 engine
                    if hasattr(worker_llm.llm_engine.engine_core, 'shutdown'):
                        worker_llm.llm_engine.engine_core.shutdown()
        except Exception as e:
            print(f"⚠️  [CLEANUP] Warning during worker model shutdown: {e}")
        finally:
            del worker_llm
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()
        
        # Small delay to ensure GPU memory is fully released and background processes are cleaned up
        # vLLM uses background processes that need time to clean up
        time.sleep(3)
        
        # Check GPU memory after cleanup
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_memory_gb = total - reserved
            print(f"✅ [CLEANUP] Models unloaded. GPU memory - Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB, Free: {free_memory_gb:.2f} GiB. Ready for next checkpoint.")
        else:
            print(f"✅ [CLEANUP] Models unloaded, GPU memory freed. Ready for next checkpoint.")


if __name__ == "__main__":

    args = parse_arguments()
    
    # Display configuration for user verification
    display_config(args)
    
    # Run the selected experiment pipeline
    if args.refiner:
        validate_refiner(
            worker_model_name=args.worker_model_name,
            refiner_model_name=args.refiner_model_name,
            dataset=args.dataset,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            max_model_len=args.max_model_len,
            number_narratives=args.number_narratives,
            num_counterfactuals_per_factual=args.num_counterfactuals_per_factual,
            fine_tuned=args.fine_tuned,
            checkpoint_every=args.checkpoint_every,
            max_checkpoint=args.max_checkpoint
        )
    else:
        validate_worker(
            model_name=args.worker_model_name,
            dataset=args.dataset,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            max_model_len=args.max_model_len,
            fine_tuned=args.fine_tuned,
            checkpoint_every=args.checkpoint_every,
            max_checkpoint=args.max_checkpoint,
            num_counterfactuals_per_factual=args.num_counterfactuals_per_factual
        )