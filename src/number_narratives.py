import os
import json
import time
import random
import re
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from data.dataset_kb import dataset_kb
from .utils import MODEL_MAPPING, prompt, get_checkpoint_step
from .functions import extract_and_parse_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def set_full_reproducibility(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_dataset_name(dataset: str) -> str:
    """Normalize dataset name for matching against JSON keys."""
    if dataset.lower() == "adult":
        return "adult income"
    if dataset.lower() == "california":
        return "california housing"
    return dataset


def _strip_reasoning_sections(text: str) -> str:
    """Remove <think>/<thinking>/<reasoning> tags from text."""
    pattern = r"<(think|thinking|reasoning)[^>]*>.*?</\\1>"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)


def _extract_explanation_from_text(text: str) -> str | None:
    """Extract explanation field from generated text using functions.extract_and_parse_json."""
    if not text:
        return None
    cleaned = _strip_reasoning_sections(text)
    parsed = extract_and_parse_json(cleaned)
    if isinstance(parsed, dict) and isinstance(parsed.get("explanation"), str):
        return parsed["explanation"]
    # Fallback regex
    m = re.search(r'"explanation"\s*:\s*"(.+?)"', cleaned, flags=re.DOTALL)
    if m:
        return m.group(1).encode("utf-8").decode("unicode_escape")
    return None


def _extract_features_importance_ranking(text: str) -> Dict[str, int] | None:
    """Extract features_importance_ranking from generated text and normalize values to integers."""
    if not text:
        return None
    cleaned = _strip_reasoning_sections(text)
    parsed = extract_and_parse_json(cleaned)
    if isinstance(parsed, dict):
        features = parsed.get("features_importance_ranking")
        if isinstance(features, dict):
            # Normalize all values to integers
            normalized = {}
            for key, value in features.items():
                try:
                    normalized[key] = int(value)
                except (ValueError, TypeError):
                    continue
            return normalized if normalized else None
    return None


def _llm_generate_vllm(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    current_prompt: str,
    lora_request: Optional[LoRARequest] = None,
    max_retries: int = 3,
) -> str:
    messages = [{"role": "user", "content": current_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            logger.debug(f"vLLM generation attempt {attempt + 1}/{max_retries}")
            with torch.no_grad():
                if lora_request:
                    outputs = llm.generate([text], sampling_params=sampling_params, lora_request=lora_request)
                else:
                    outputs = llm.generate([text], sampling_params=sampling_params)
            for output in outputs:
                generated_text = output.outputs[0].text
                logger.debug(f"vLLM generation successful (length: {len(generated_text)} chars)")
                return generated_text
        except AssertionError as e:
            last_err = e
            logger.warning(f"vLLM generation attempt {attempt + 1} failed (AssertionError): {e}")
        except Exception as e:
            last_err = e
            logger.warning(f"vLLM generation attempt {attempt + 1} failed (Exception): {e}")
    raise RuntimeError(f"vLLM generation failed after {max_retries} attempts: {last_err}")


def assess_narratives(
    model_name: str,
    temperature: float,
    top_p: float,
    dataset: str,
    max_tokens: int,
    repetition_penalty: float,
    max_model_len: int,
    num_narratives: int = 8,
) -> None:
    """
    Generate K narratives per sample across the validation set.
    Saves raw responses to results/number_narratives/{dataset}/{model_name}.json
    """
    logger.info("=" * 60)
    logger.info(f"Starting narrative generation for model: {model_name}, dataset: {dataset}")
    logger.info("=" * 60)
    
    set_full_reproducibility()

    # Validation set configuration (like pipeline.py)
    NUM_FACTUALS = 10
    NUM_COUNTERFACTUALS_PER_FACTUAL = 2
    LOWER_BOUND = 0
    UPPER_BOUND = NUM_FACTUALS * NUM_COUNTERFACTUALS_PER_FACTUAL - 1  # 19 (0-19 = 20 samples)
    
    logger.info(f"Processing {NUM_FACTUALS} factuals with {NUM_COUNTERFACTUALS_PER_FACTUAL} counterfactuals each")
    logger.info(f"Total samples: {NUM_FACTUALS * NUM_COUNTERFACTUALS_PER_FACTUAL} (range [{LOWER_BOUND}, {UPPER_BOUND}])")
    logger.info(f"Generating {num_narratives} narratives per sample")

    # Normalize dataset name for matching JSON keys
    dataset_norm = _normalize_dataset_name(dataset)

    # Prepare output directory & file
    out_dir = os.path.join("results", "number_narratives", dataset)
    _ensure_outdir(out_dir)
    out_json = os.path.join(out_dir, f"{model_name}.json")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Output JSON file: {out_json}")

    # Load validation counterfactuals data (like validation.py)
    logger.info("Loading validation counterfactuals data...")
    with open(os.path.join("src", "explainer", "val_counterfactuals.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded data for {len(data)} datasets")

    # Build prompt base
    base_prompt = prompt

    # Resolve model path
    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Unknown model '{model_name}'. Not found in MODEL_MAPPING.")
    
    resolved_model = MODEL_MAPPING[model_name]
    logger.info(f"Using vLLM model from MODEL_MAPPING: {resolved_model}")
    
    # Get LoRA checkpoint
    checkpoint_step = get_checkpoint_step(dataset, "draft_generator", model_name, default=500)
    lora_checkpoint_path = f"outputs_unsloth/outputs_unsloth_{dataset}_worker/{model_name}/checkpoint-{checkpoint_step}"
    logger.info(f"LoRA checkpoint path: {lora_checkpoint_path}")
    
    # Check if LoRA checkpoint exists
    if not os.path.exists(lora_checkpoint_path):
        raise ValueError(f"LoRA checkpoint not found at: {lora_checkpoint_path}")
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    
    logger.info("Creating sampling parameters...")
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        top_k=10,
        stop=tokenizer.eos_token,
    )
    
    logger.info(f"Loading vLLM model with LoRA support (this may take a while)...")
    llm_start = time.time()
    vllm_llm = LLM(
        model=resolved_model,
        gpu_memory_utilization=0.5,
        max_model_len=max_model_len,
        max_num_seqs=1,
        enable_lora=True,
    )
    llm_time = time.time() - llm_start
    logger.info(f"vLLM model loaded in {llm_time:.2f}s")
    
    # Create LoRA request
    lora_request = LoRARequest(
        "counterfactual_explainability_adapter",
        1,
        lora_checkpoint_path,
    )

    results: Dict[str, Any] = {}
    sample_counter = 0

    # Iterate like pipeline.py
    for dataset_name, examples in data.items():
        # Match dataset name (case-insensitive, like pipeline.py)
        if dataset_name.lower() != dataset_norm:
            continue
        
        # Sort indices to ensure consistent ordering (indices are strings like "0", "1", etc.)
        sorted_indices = sorted(examples.keys(), key=lambda x: int(x))[:NUM_FACTUALS]
        
        for index in sorted_indices:
            values = examples[index]
            # Process only the first NUM_COUNTERFACTUALS_PER_FACTUAL counterfactuals
            counterfactuals_to_process = values["counterfactuals"][:NUM_COUNTERFACTUALS_PER_FACTUAL]
            
            for counterfactual in counterfactuals_to_process:
                if LOWER_BOUND <= sample_counter <= UPPER_BOUND:
                    current_prompt = base_prompt.format(
                        dataset_description=dataset_kb[dataset_name],
                        factual_example=str(values["factual"]),
                        counterfactual_example=str(counterfactual),
                    )

                    logger.info(f"Processing sample {sample_counter} (dataset: {dataset_name}, index: {index})")
                    logger.info(f"Generating {num_narratives} narratives for sample {sample_counter}...")
                    narr_start_time = time.time()
                    
                    # Generate narratives with retry logic
                    narratives: List[str] = []
                    features_importance_ranking_lists: List[Dict[str, int]] = []
                    
                    for j in range(num_narratives):
                        logger.info(f"  Generating narrative {j+1}/{num_narratives} for sample {sample_counter}...")
                        narr_gen_start = time.time()
                        
                        gen_text = _llm_generate_vllm(
                            vllm_llm, tokenizer, sampling_params, current_prompt, 
                            lora_request=lora_request, max_retries=3
                        )
                        
                        narr_gen_time = time.time() - narr_gen_start
                        logger.info(f"  Narrative {j+1}/{num_narratives} generated in {narr_gen_time:.2f}s")
                        
                        logger.debug(f"  Extracting explanation and features_importance_ranking from generated text...")
                        explanation = _extract_explanation_from_text(gen_text)
                        
                        if explanation is None:
                            logger.warning(f"  Failed to extract explanation on first attempt, retrying...")
                            # Treat extraction failure as failure and retry up to 2 more times
                            retry_ok = False
                            for retry_num in range(2):
                                logger.info(f"  Retry {retry_num+1}/2 for narrative {j+1}/{num_narratives}...")
                                gen_text = _llm_generate_vllm(
                                    vllm_llm, tokenizer, sampling_params, current_prompt,
                                    lora_request=lora_request, max_retries=1
                                )
                                explanation = _extract_explanation_from_text(gen_text)
                                if explanation is not None:
                                    retry_ok = True
                                    logger.info(f"  Successfully extracted explanation on retry {retry_num+1}")
                                    break
                            if not retry_ok:
                                logger.warning(f"  Failed to extract explanation after retries, continuing...")
                                continue
                        else:
                            logger.debug(f"  Explanation extracted successfully (length: {len(explanation)} chars)")
                        
                        # Store narrative explanation
                        narratives.append(explanation)

                        # Extract features_importance_ranking (treat missing as empty dict)
                        features_importance_ranking = _extract_features_importance_ranking(gen_text)
                        features_importance_ranking_lists.append(features_importance_ranking if features_importance_ranking else {})
                        num_features = len(features_importance_ranking) if features_importance_ranking else 0
                        logger.info(f"  Narrative {j+1}/{num_narratives} completed (features: {num_features} items)")

                    narr_total_time = time.time() - narr_start_time
                    logger.info(f"All {num_narratives} narratives generated for sample {sample_counter} in {narr_total_time:.2f}s")

                    # Store per-sample (raw responses only, no metrics computation)
                    results[str(sample_counter)] = {
                        "dataset": dataset_name,
                        "index": index,
                        "factual": values["factual"],
                        "counterfactual": counterfactual,
                        "narratives": narratives,
                        "features_importance_ranking": features_importance_ranking_lists,
                    }

                    # Incremental save every 10 samples
                    if sample_counter % 10 == 0:
                        logger.info(f"Saving intermediate results (sample {sample_counter})...")
                        with open(out_json, "w", encoding="utf-8") as out_f:
                            json.dump(results, out_f, indent=2, ensure_ascii=False)
                        logger.info(f"Intermediate results saved to {out_json}")

                sample_counter += 1

    logger.info(f"Processing complete. Processed {sample_counter} samples.")
    
    # Final save
    logger.info("Saving final results...")
    with open(out_json, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=2, ensure_ascii=False)
    logger.info(f"Final results saved to {out_json}")
    
    logger.info("=" * 60)
    logger.info("Narrative generation completed successfully!")
    logger.info("=" * 60)


__all__ = ["assess_narratives"]
