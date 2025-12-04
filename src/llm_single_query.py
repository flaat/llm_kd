from transformers import AutoTokenizer
import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import random
import numpy as np
import time
import os
from utils import MODEL_MAPPING
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

def load_model(model_name, temperature, top_p, repetition_penalty, max_tokens, max_model_len):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        repetition_penalty=repetition_penalty, 
        max_tokens=max_tokens, 
        top_k=10,
        stop=tokenizer.eos_token
    )

    llm = LLM(
        model=model_name, 
        gpu_memory_utilization=0.95, 
        max_model_len=max_model_len, 
        max_num_seqs=1
    )

    return llm, tokenizer, sampling_params

def llm_call(llm, tokenizer, sampling_params, prompt):

    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    try:
        with torch.no_grad():
            response = llm.generate([text], sampling_params=sampling_params)

            for output in response:
                generated_text = output.outputs[0].text
            
            return generated_text

    except AssertionError as assert_e:
        print(f"🚨 Assertion error: {assert_e}")
        return assert_e
    

def main():
    #set_full_reproducibility(42)
    model_name = "qwen3_32B"  # Change this to select a different model
    hf_model_name = MODEL_MAPPING[model_name]

    temperature = 0.6
    top_p = 0.8
    repetition_penalty = 1.1
    max_tokens = 2048
    max_model_len = 2048

    llm, tokenizer, sampling_params = load_model(hf_model_name, temperature, top_p, repetition_penalty, max_tokens, max_model_len)

    prompt = "What is the capital of France?"

    print(f"*** Prompt: {prompt}\n")
    response = llm_call(llm, tokenizer, sampling_params, prompt)
    print(f"*** Response: {response}\n")
    print("*** Done.")












if __name__ == "__main__":
    main()