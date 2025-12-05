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
from src.utils import MODEL_MAPPING, prompt, prompt_ref
import re
from vllm.lora.request import LoRARequest
import argparse

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
    print("\n----- Experiment Settings -----")
    print(f"Dataset:             {args.dataset}")
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
        default=5500,
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
        default=50,
        help='Number of iterations between saving checkpoints (default: 50)'
    )

    parser.add_argument(
        '--max_checkpoint',
        type=int,
        default=1342,
        help='Maximum number of checkpoints to save (default: 1342)'
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



def validate_worker(model_name: str, dataset: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float, max_model_len, fine_tuned=True, checkpoint_every=50, max_checkpoint=1342):

    print(f"Params list: {model_name}, {temperature}, {top_p}, {max_tokens}, {repetition_penalty}, {max_model_len}, {fine_tuned}, {checkpoint_every}, {max_checkpoint}")
    set_full_reproducibility()

    # Check if the checkpoint directory exists
    lora_max_checkpoint_directory_path = f"outputs_unsloth/outputs_unsloth_{dataset}_worker/{model_name}/checkpoint-{max_checkpoint}"
    if not os.path.exists(lora_max_checkpoint_directory_path):
        raise ValueError(f"Max checkpoint value wrong: {lora_max_checkpoint_directory_path} does not exist")
    
    LOWER_BOUND = 0
    UPPER_BOUND = 19
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

    checkpoint_steps = list(range(checkpoint_every, max_checkpoint + 1, checkpoint_every))
    if checkpoint_steps[-1] != max_checkpoint:
        checkpoint_steps.append(max_checkpoint)

    old_dataset_name = dataset
    
    for checkpoint in checkpoint_steps:

        print(f"🔄 Loading checkpoint {checkpoint}...")
        lora_checkpoint_directory_path = f"outputs_unsloth/outputs_unsloth_{old_dataset_name}_worker/{name}/checkpoint-{checkpoint}"
        
        # Initialize LLM with optimized GPU memory usage
        if fine_tuned:
            llm = LLM(
            model=model_name, 
            gpu_memory_utilization=0.7, 
            max_model_len=max_model_len, 
            max_num_seqs=1,
            enable_lora=True,
            trust_remote_code=True
        )
        else:
            llm = LLM(
                model=model_name, 
                gpu_memory_utilization=0.7, 
                max_model_len=max_model_len, 
                max_num_seqs=1
            )

        # Define output file for results
        responses = {}  # Dictionary to store new responses
        # Load counterfactual data
        with open(f"src/explainer/val_counterfactuals.json", 'r', encoding='utf-8') as file1:
            data = json.load(file1)

        i = 0  # Counter for responses

        for dataset_name, examples in data.items():
            if dataset == "adult":
                dataset = "adult income"

            if dataset_name.lower() == dataset:
                output_directory = f"results/fine-tuning/worker_validation/{old_dataset_name}/{name}"
                os.makedirs(output_directory, exist_ok=True)
                output_file = f"{output_directory}/{name}_checkpoint_{checkpoint}.json"

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
                            try:
                                
                                with torch.no_grad():
                                    start_time = time.time()
                                    if fine_tuned:
                                        outputs = llm.generate([text], sampling_params=sampling_params, lora_request=LoRARequest("counterfactual_explainability_adapter", 1, lora_checkpoint_directory_path))
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
        del llm
        torch.cuda.empty_cache()


if __name__ == "__main__":

    args = parse_arguments()
    
    # Display configuration for user verification
    display_config(args)
    
    # Run the selected experiment pipeline
    if args.refiner:
            print("Refiner validation not implemented yet.")    
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
            max_checkpoint=args.max_checkpoint
        )