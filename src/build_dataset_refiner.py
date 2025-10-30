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

def extract_narratives(results: list, number_narratives: int):
    """
    Extracts explanations from the generated outputs.
    """
    import ast

    def _strip_reasoning_sections(text: str) -> str:
        # Rimuove blocchi di ragionamento: <think>...</think>, <thinking>...</thinking>, <reasoning>...</reasoning>
        pattern = r"<(think|thinking|reasoning)[^>]*>.*?</\1>"
        return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    def _last_code_block_json(text: str):
        # Cerca l’ultimo blocco ```json ... ``` oppure l’ultimo ``` ... ```
        matches = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if not matches:
            matches = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
        return matches[-1].strip() if matches else None

    def _last_balanced_json_object(text: str):
        # Trova l’ultimo oggetto JSON bilanciato scorrendo le graffe
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
        # Tenta json standard, poi fallback a ast.literal_eval (apici singoli)
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(blob)
                # Converte in JSON-compatibile (chiavi/valori stringa devono essere str)
                if isinstance(obj, (dict, list)):
                    return json.loads(json.dumps(obj))
            except Exception:
                return None
        except Exception:
            return None

    narratives = []
    for outputs in results:
        text = None
        # Prendi il primo output disponibile
        for output in outputs:
            text = output.outputs[0].text
            break

        if not text:
            print("⚠️ Nessun testo generato nell'output.")
            narratives.append(None)
            continue

        # 1) Rimuovi le sezioni di thinking prima di cercare il JSON finale
        cleaned = _strip_reasoning_sections(text)

        response = None

        # 2) Preferisci un blocco ```json ... ```
        blob = _last_code_block_json(cleaned)
        if blob:
            response = _parse_json_loose(blob)

        # 3) In assenza, prova a prendere l’ultimo oggetto JSON bilanciato
        if response is None:
            blob = _last_balanced_json_object(cleaned)
            if blob:
                response = _parse_json_loose(blob)

        # 4) Se trovato, estrai la narrative
        if isinstance(response, dict):
            if "explanation" in response and isinstance(response["explanation"], str):
                narratives.append(response["explanation"])
                continue
            else:
                print(f"⚠️ Chiave 'explanation' non trovata. Chiavi disponibili: {list(response.keys())}")
                narratives.append(None)
                continue

        # 5) Fallback: prova regex mirata all'attributo explanation in JSON-like
        m = re.search(r'"explanation"\s*:\s*"(.+?)"', cleaned, flags=re.DOTALL)
        if m:
            # De-escape basilare
            explanation = m.group(1).encode("utf-8").decode("unicode_escape")
            narratives.append(explanation)
        else:
            print(f"⚠️ Nessun JSON valido trovato. Estratto (inizio): {cleaned[:200]}...")
            narratives.append(None)

    # Padding/trim alla dimensione richiesta
    return tuple(narratives[:number_narratives] + [None] * (number_narratives - len(narratives)))


def build_dataset(worker_model_name: str, refiner_model_name: str, temperature: float, top_p: float, dataset: str, max_tokens: int, repetition_penalty: float, max_model_len, number_narratives: int):
    set_full_reproducibility()
    
    LOWER_BOUND = 0
    UPPER_BOUND = 0
    
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


    # Define output file for results
    output_file = f"data/{dataset}_Refiner_{worker_model_name}_{refiner_model_name}.json"
    responses = {}  # Dictionary to store new responses

    # Load counterfactual data
    with open(f"src/explainer/counterfactuals.json", 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    
    i = 0  # Counter for responses

    if dataset == "adult":
        dataset = "adult income"

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
                        
                        # Load worker model
                        print(f"🔧 Loading Worker Model: {worker_model_name}")
                        worker_llm = LLM(
                            model=worker_model_path, 
                            gpu_memory_utilization=0.92, 
                            max_model_len=max_model_len, 
                            max_num_seqs=1
                        )
                        
                        # Generate draft narratives using the worker LLM
                        N = number_narratives
                        narratives = []
                        for j in range(N):
                            try:
                                with torch.no_grad():
                                    print(f"🔄 Generating draft narrative {j+1} of {N} using worker model")
                                    draft_start = time.time()
                                    outputs = worker_llm.generate([text_worker], sampling_params=sampling_params)
                                    # Process the outputs if generated successfully
                                    for output in outputs:
                                        prompt = output.prompt
                                        generated_text = output.outputs[0].text
                                    print(generated_text)
                                    narratives.append(outputs)
                                    draft_end = time.time()
                                    print(f"⏱️  Draft narrative {j+1} generated in {draft_end - draft_start:.2f}s")
                            except AssertionError as assert_e:
                                print(f"🚨 Assertion error: {assert_e}")
                                continue
                        
                        narratives = extract_narratives(narratives, number_narratives)
                        
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
                                outputs = refiner_llm.generate([text_refiner], sampling_params=sampling_params)
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

