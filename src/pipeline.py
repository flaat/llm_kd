from transformers import AutoTokenizer
import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import random
import numpy as np
import time
import os
from data.dataset_kb import dataset_kb
from .utils import MODEL_MAPPING, prompt, prompt_ref
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


def test_llm(model_name: str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float, max_model_len, fine_tuned=False):
    
    print(f"Params list: {model_name}, {temperature}, {top_p}, {max_tokens}, {repetition_penalty}, {max_model_len}, {fine_tuned}")
    set_full_reproducibility()
    
    LOWER_BOUND = 1
    UPPER_BOUND = 200
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
        gpu_memory_utilization=0.8, 
        max_model_len=max_model_len, 
        max_num_seqs=1,
        enable_lora=True
    )
    else:
        llm = LLM(
            model=model_name, 
            gpu_memory_utilization=0.8, 
            max_model_len=max_model_len, 
            max_num_seqs=1
        )


    lora_checkpoint_directory_path = f"outputs_unsloth_titanic/{name}/checkpoint-500"

    # Define output file for results
    responses = {}  # Dictionary to store new responses

    # Load counterfactual data
    with open(f"src/explainer/test_counterfactuals.json", 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    i = 0  # Counter for responses

    for dataset_name, examples in data1.items():
        if dataset_name == "Titanic":
            output_file = f"data/results/{model_name.split('/')[-1]}_Response_{dataset_name}_Finetuned_{fine_tuned}.json"

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
                                start = time.time()
                                
                                if fine_tuned:
                                    outputs = llm.generate([text], sampling_params=sampling_params, lora_request=LoRARequest("counterfactual_explainability_adapter", 1, lora_checkpoint_directory_path))
                                else:
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
                        responses[i] = {"generated_text": generated_text, "prompt": prompt, "ground_truth": {"counterfactual":counterfactual, "factual": values["factual"]}, "changes": extract_feature_changes(values["factual"], counterfactual)}
                        

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



def test_llm_refiner(worker_model_name: str, refiner_model_name: str, dataset:str, temperature: float, top_p: float, max_tokens: int, repetition_penalty: float, max_model_len, fine_tuned=False):

    print(f"Params list: {worker_model_name}, {refiner_model_name}, {temperature}, {top_p}, {max_tokens}, {repetition_penalty}, {max_model_len}, {fine_tuned}")
    set_full_reproducibility()
    
    LOWER_BOUND = 1
    UPPER_BOUND = 200
    worker_name = worker_model_name
    refiner_name = refiner_model_name
    global prompt
    global prompt_ref
    base_prompt = prompt
    base_prompt_ref = prompt_ref
    worker_model_name = MODEL_MAPPING[worker_model_name]
    refiner_model_name = MODEL_MAPPING[refiner_model_name]

    tokenizer_worker = AutoTokenizer.from_pretrained(worker_model_name)
    tokenizer_refiner = AutoTokenizer.from_pretrained(refiner_model_name)
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

    lora_checkpoint_directory_path_worker = f"outputs_unsloth_{dataset}/{worker_name}/checkpoint-500"
    lora_checkpoint_directory_path_refiner = f"outputs_unsloth_{dataset}_refiner/{refiner_name}/checkpoint-800"

    # Define output file for results
    responses = {}  # Dictionary to store new responses

    # Load counterfactual data
    with open(f"src/explainer/test_counterfactuals.json", 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    i = 0  # Counter for responses
    output_file = f"data/results/Worker_{worker_model_name.split('/')[-1]}_Refiner_{refiner_model_name.split('/')[-1]}_Response_{dataset}_Finetuned_{fine_tuned}.json"

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
                        # Tokenize the messages for the worker LLM
                        text = tokenizer_worker.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        worker_llm = LLM(
                            model=worker_model_name, 
                            gpu_memory_utilization=0.85, 
                            max_model_len=max_model_len, 
                            max_num_seqs=1,
                            enable_lora=True
                        )
                        # Generate explanations using the worker LLM
                        N = 3
                        explanations = []
                        for _ in range(N):
                            try:
                                with torch.no_grad():
                                    #start = time.time()
                                    outputs = worker_llm.generate([text], sampling_params=sampling_params_worker, lora_request=LoRARequest("counterfactual_explainability_adapter_worker", 1, lora_checkpoint_directory_path_worker))
                                    explanations.append(outputs)
                                    #end = time.time()
                            except AssertionError as assert_e:
                                print(f"🚨 Assertion error: {assert_e}")
                                continue
                        
                        explanation1, explanation2, explanation3 = extract_explanations(explanations)

                        # Delete the worker LLM to free memory
                        del worker_llm
                        torch.cuda.empty_cache()

                        current_prompt_refiner = base_prompt_ref.format(
                            dataset_description=dataset_kb[dataset_name], 
                            factual_example=str(values["factual"]), 
                            counterfactual_example=str(counterfactual),
                            draft_explanation_1=explanation1,
                            draft_explanation_2=explanation2,
                            draft_explanation_3=explanation3
                        )

                        messages = [{"role": "user", "content": current_prompt_refiner}]

                        text = tokenizer_refiner.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )

                        if fine_tuned:
                            refiner_llm = LLM(
                                model=refiner_model_name, 
                                gpu_memory_utilization=0.85, 
                                max_model_len=max_model_len, 
                                max_num_seqs=1,
                                enable_lora=True
                            )
                        else:
                            refiner_llm = LLM(
                                model=refiner_model_name, 
                                gpu_memory_utilization=0.85, 
                                max_model_len=max_model_len, 
                                max_num_seqs=1
                            )

                        try:
                            with torch.no_grad():
                                if fine_tuned:
                                    start = time.time()
                                    outputs = refiner_llm.generate([text], sampling_params=sampling_params_refiner, lora_request=LoRARequest("counterfactual_explainability_adapter_refiner", 2, lora_checkpoint_directory_path_refiner))
                                    end = time.time()
                                else:
                                    start = time.time()
                                    outputs = refiner_llm.generate([text], sampling_params=sampling_params_refiner)
                                    end = time.time()
                        except AssertionError as assert_e:
                            print(f"🚨 Assertion error: {assert_e}")
                            continue

                        # Delete the refiner LLM to free memory
                        del refiner_llm
                        torch.cuda.empty_cache()

                        # Process the outputs if generated successfully
                        for output in outputs:
                            prompt = output.prompt
                            generated_text = output.outputs[0].text

                        print(generated_text)
                        responses[i] = {"generated_text": generated_text, "prompt": prompt, "ground_truth": {"counterfactual":counterfactual, "factual": values["factual"]}, "changes": extract_feature_changes(values["factual"], counterfactual)}
                        

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