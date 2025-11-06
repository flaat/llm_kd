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
from transformers import AutoTokenizer

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


def build_dataset_wor(model_name: str, temperature: float, top_p: float, dataset: str, max_tokens: int, repetition_penalty: float, max_model_len):
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

    # Define output file for results
    output_file = f"data/{dataset}_Worker_{model_name}.json"
    responses = {}  # Dictionary to store new responses

    # Load counterfactual data
    with open(f"src/explainer/counterfactuals.json", 'r', encoding='utf-8') as file1:
        data = json.load(file1)

    
    i = 0  # Counter for responses

    if dataset == "adult":
        dataset = "adult income"

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
                        
                        # Start timing for the entire process 
                        total_start_time = time.time()
                        
                        try:
                            with torch.no_grad():
                                outputs = llm.generate([text], sampling_params=sampling_params)
                        except AssertionError as assert_e:
                            print(f"🚨 Assertion error: {assert_e}")
                            continue
                        
                        
                        # Calculate total time for draft generation + refinement
                        total_end_time = time.time()
                        total_time = total_end_time - total_start_time
                        
                        # Process the outputs if generated successfully
                        for output in outputs:
                            prompt = output.prompt
                            generated_text = output.outputs[0].text

                        print(generated_text)
                        responses[i] = {"prompt": prompt, "generated_text": generated_text}
                        

                        print(f"#################### Total time taken: {total_time:.2f} seconds, explanation number {i} ###########################")

                        # Save every 10 responses
                        if i % 10 == 0:
                            save_responses(responses, output_file)
                            responses = {}  # Clear the temporary dictionary to prevent duplication

                        # Delete large variables to free memory
                        del generated_text
                        del outputs
                    i += 1
    # Final save after loop completion
    save_responses(responses, output_file)

