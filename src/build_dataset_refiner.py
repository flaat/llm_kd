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
            return explanations.append(None)

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


def build_dataset(model_name: str, temperature: float, top_p: float, dataset: str, max_tokens: int, repetition_penalty: float, max_model_len):
    set_full_reproducibility()
    
    LOWER_BOUND = 0
    UPPER_BOUND = 50000
    
    global prompt
    global prompt_ref
    base_prompt = prompt
    base_prompt_ref = prompt_ref
    model_name = MODEL_MAPPING[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        repetition_penalty=repetition_penalty, 
        max_tokens=max_tokens, 
        top_k=10
    )

    # Initialize LLM with optimized GPU memory usage
    llm = LLM(
        model=model_name, 
        gpu_memory_utilization=0.92, 
        max_model_len=max_model_len, 
        max_num_seqs=1
    )

    # Initialize worker LLM for generating explanations
    # Using a fine-tuned smaller model
    worker_llm = LLM(
        model="unsloth/Qwen2.5-0.5B-Instruct", 
        gpu_memory_utilization=0.97, 
        max_model_len=max_model_len, 
        max_num_seqs=1,
        enable_lora=True
    )

    # Define output file for results
    output_file = f"data/{model_name.split('/')[-1]}_Refiner.json"
    responses = {}  # Dictionary to store new responses

    # Load counterfactual data
    with open(f"src/explainer/counterfactuals.json", 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    
    from vllm.lora.request import LoRARequest

    lora_checkpoint_directory_path = f"outputs_unsloth/unsloth_qwen_0.5B/checkpoint-500"

    i = 0  # Counter for responses

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
                        
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        # Generate explanations using the worker LLM
                        N = 3
                        explanations = []
                        for j in range(N):
                            try:
                                with torch.no_grad():
                                    start = time.time()
                                    outputs = worker_llm.generate([text], sampling_params=sampling_params, lora_request=LoRARequest("counterfactual_explainability_adapter", 1, lora_checkpoint_directory_path))
                                    explanations.append(outputs)
                                    end = time.time()
                            except AssertionError as assert_e:
                                print(f"🚨 Assertion error: {assert_e}")
                                continue
                        
                        explanation1, explanation2, explanation3 = extract_explanations(explanations)

                        current_prompt_refiner = base_prompt_ref.format(
                            dataset_description=dataset_kb[dataset_name], 
                            factual_example=str(values["factual"]), 
                            counterfactual_example=str(counterfactual),
                            draft_explanation_1=explanation1,
                            draft_explanation_2=explanation2,
                            draft_explanation_3=explanation3
                        )

                        messages = [{"role": "user", "content": current_prompt_refiner}]

                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )

                        try:
                            with torch.no_grad():
                                start = time.time()
                                outputs = llm.generate([text], sampling_params=sampling_params)
                                end = time.time()
                        except AssertionError as assert_e:
                            print(f"🚨 Assertion error: {assert_e}")
                            continue
                        
                        # Process the outputs if generated successfully
                        for output in outputs:
                            prompt = output.prompt
                            generated_text = output.outputs[0].text

                        print(generated_text)
                        responses[i] = {"prompt": prompt, "generated_text": generated_text}
                        

                        print(f"#################### Time taken: {end - start:.2f} seconds, explanation number {i} ###########################")

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

"""
<user>input</user>

<thinking>thinking</thinking><assistant>outtput</assistant>

[...] similarly to Deepseek R1 (citazione), we train our model using an RL-free Inference-time Compute objective [...]

"""

