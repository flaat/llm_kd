import os
import json
import time
import random
import re
import logging
from typing import Any, Dict, List
from google import genai
import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from scipy.stats import kendalltau

from data.dataset_kb import dataset_kb
from .utils import MODEL_MAPPING, GOOGLE_API_MODEL_MAPPING, prompt

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
    if dataset.lower() == "adult":
        return "adult income"
    return dataset


def _strip_reasoning_sections(text: str) -> str:
    pattern = r"<(think|thinking|reasoning)[^>]*>.*?</\\1>"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)


def _last_code_block_json(text: str) -> str | None:
    matches = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        matches = re.findall(r"```(.*?)```", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def _last_balanced_json_object(text: str) -> str | None:
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


def _parse_json_loose(blob: str) -> Dict[str, Any] | None:
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        try:
            import ast
            obj = ast.literal_eval(blob)
            if isinstance(obj, (dict, list)):
                return json.loads(json.dumps(obj))
        except Exception:
            return None
    except Exception:
        return None


def _extract_explanation_from_text(text: str) -> str | None:
    if not text:
        return None
    cleaned = _strip_reasoning_sections(text)
    blob = _last_code_block_json(cleaned)
    response = _parse_json_loose(blob) if blob else None
    if response is None:
        blob2 = _last_balanced_json_object(cleaned)
        response = _parse_json_loose(blob2) if blob2 else None
    if isinstance(response, dict) and isinstance(response.get("explanation"), str):
        return response["explanation"]
    m = re.search(r'"explanation"\s*:\s*"(.+?)"', cleaned, flags=re.DOTALL)
    if m:
        return m.group(1).encode("utf-8").decode("unicode_escape")
    return None


def _extract_features_importance_ranking(text: str) -> Dict[str, int] | None:
    """Extract features_importance_ranking dictionary from generated text and normalize values to integers."""
    if not text:
        return None
    cleaned = _strip_reasoning_sections(text)
    blob = _last_code_block_json(cleaned)
    response = _parse_json_loose(blob) if blob else None
    if response is None:
        blob2 = _last_balanced_json_object(cleaned)
        response = _parse_json_loose(blob2) if blob2 else None
    if isinstance(response, dict):
        features = response.get("features_importance_ranking")
        if isinstance(features, dict):
            # Normalize all values to integers
            normalized = {}
            for key, value in features.items():
                try:
                    normalized[key] = int(value)
                except (ValueError, TypeError):
                    # Skip invalid values
                    continue
            return normalized if normalized else None
    # Try regex fallback for features_importance_ranking
    m = re.search(r'"features_importance_ranking"\s*:\s*\{([^}]+)\}', cleaned, flags=re.DOTALL)
    if m:
        try:
            # Try to extract key-value pairs
            content = m.group(1)
            # Match "key": "value" patterns (both quoted and unquoted values)
            # First try quoted values
            pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', content)
            if not pairs:
                # Try unquoted numeric values
                pairs = re.findall(r'"([^"]+)"\s*:\s*(\d+)', content)
            if pairs:
                normalized = {}
                for key, value in pairs:
                    try:
                        normalized[key] = int(value)
                    except (ValueError, TypeError):
                        continue
                return normalized if normalized else None
        except Exception:
            pass
    return None


def _llm_generate_vllm(
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    current_prompt: str,
    max_retries: int = 3,
) -> str:
    messages = [{"role": "user", "content": current_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            logger.debug(f"vLLM generation attempt {attempt + 1}/{max_retries}")
            with torch.no_grad():
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


def _llm_generate_google(
    client: Any,
    model_name: str,
    current_prompt: str,
    max_retries: int = 3,
    backoff_seconds: int = 5,
    temperature: float = 0.6,
    top_p: float = 0.8
) -> str:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            logger.debug(f"Google API generation attempt {attempt + 1}/{max_retries} (model: {model_name})")
            # NOTE:
            # - Use 'config=' parameter with GenerateContentConfig instance.
            # - Field names use snake_case (temperature, top_p) in GenerateContentConfig.
            #     temperature: float in [0.0, 2.0]
            #     top_p: float in [0.0, 1.0]
            generation_config = genai.types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
            )
            response = client.models.generate_content(
                model=model_name, 
                contents=current_prompt,
                config=generation_config,
            )
            generated_text = getattr(response, "text", None) or str(response)
            logger.debug(f"Google API generation successful (length: {len(generated_text)} chars)")
            return generated_text
        except Exception as e:
            last_err = e
            logger.warning(f"Google API generation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Waiting {backoff_seconds} seconds before retry...")
                time.sleep(backoff_seconds)
    raise RuntimeError(f"Google generation failed after {max_retries} attempts: {last_err}")


def _compute_jaccard_similarity(dict1: Dict[str, int], dict2: Dict[str, int]) -> float:
    """Compute Jaccard similarity between two dictionaries based on their keys (feature names)."""
    set1 = set(dict1.keys()) if dict1 else set()
    set2 = set(dict2.keys()) if dict2 else set()
    # Handle empty dicts: two empty dicts have similarity 1.0, empty and non-empty have 0.0
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _compute_kendall_tau_pairwise(dict1: Dict[str, int], dict2: Dict[str, int]) -> float:
    """Compute Kendall's tau between two features_importance_ranking dictionaries on intersecting features."""
    if not dict1 or not dict2:
        return float("nan")
    
    # Get intersection of features
    common_features = set(dict1.keys()) & set(dict2.keys())
    if len(common_features) < 2:
        # Need at least 2 features to compute Kendall's tau
        return float("nan")
    
    # Create ordered lists of ranks (values are already integers)
    # Use consistent ordering (sorted by feature name) for both dictionaries
    try:
        ranks1 = []
        ranks2 = []
        for feature in sorted(common_features):  # Sort for consistent ordering
            rank1 = dict1[feature]
            rank2 = dict2[feature]
            # Values should already be integers, but handle edge cases
            if isinstance(rank1, (int, float)) and isinstance(rank2, (int, float)):
                ranks1.append(int(rank1))
                ranks2.append(int(rank2))
            else:
                # Skip if rank is not a valid number
                continue
        
        if len(ranks1) < 2:
            return float("nan")
        
        # Compute Kendall's tau on the rank sequences
        tau, _ = kendalltau(ranks1, ranks2)
        return float(tau) if not np.isnan(tau) else float("nan")
    except Exception as e:
        logger.debug(f"Error computing Kendall's tau: {e}")
        return float("nan")


def _compute_avg_pairwise_jaccard_similarity(features_importance_ranking_lists: List[Dict[str, int]], n: int) -> float:
    """Compute average pairwise Jaccard similarity for first n features_importance_ranking dictionaries."""
    if len(features_importance_ranking_lists) < n:
        logger.debug(
            f"Requested n={n} narratives but only {len(features_importance_ranking_lists)} available; skipping Jaccard computation."
        )
        return float("nan")
    subset = features_importance_ranking_lists[:n]
    if len(subset) < 2:
        logger.debug(f"Not enough dictionaries for pairwise comparison (n={n}, len={len(subset)})")
        return float("nan")
    
    logger.debug(f"Computing Jaccard similarities for {len(subset)} feature dictionaries (n={n})")
    start_time = time.time()
    
    # Compute pairwise Jaccard similarities
    similarities = []
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            # Handle None values (missing extraction)
            dict1 = subset[i] if subset[i] else {}
            dict2 = subset[j] if subset[j] else {}
            sim = _compute_jaccard_similarity(dict1, dict2)
            similarities.append(sim)
    
    comp_time = time.time() - start_time
    avg_sim = float(np.mean(similarities)) if similarities else float("nan")
    logger.debug(f"Jaccard similarity matrix computed in {comp_time:.2f}s ({len(similarities)} pairs)")
    logger.debug(f"Average pairwise Jaccard similarity for n={n}: {avg_sim:.4f} (from {len(similarities)} pairs)")
    return avg_sim


def _compute_avg_pairwise_kendall_tau(features_importance_ranking_lists: List[Dict[str, int]], n: int) -> float:
    """Compute average pairwise Kendall's tau for first n features_importance_ranking dictionaries."""
    if len(features_importance_ranking_lists) < n:
        logger.debug(
            f"Requested n={n} narratives but only {len(features_importance_ranking_lists)} available; skipping Kendall computation."
        )
        return float("nan")
    subset = features_importance_ranking_lists[:n]
    if len(subset) < 2:
        logger.debug(f"Not enough dictionaries for pairwise comparison (n={n}, len={len(subset)})")
        return float("nan")
    
    logger.debug(f"Computing Kendall's tau for {len(subset)} feature dictionaries (n={n})")
    start_time = time.time()
    
    # Compute pairwise Kendall's tau
    taus = []
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            # Handle None values (missing extraction)
            dict1 = subset[i] if subset[i] else {}
            dict2 = subset[j] if subset[j] else {}
            tau = _compute_kendall_tau_pairwise(dict1, dict2)
            if not np.isnan(tau):
                taus.append(tau)
    
    comp_time = time.time() - start_time
    avg_tau = float(np.mean(taus)) if taus else float("nan")
    logger.debug(f"Kendall's tau matrix computed in {comp_time:.2f}s ({len(taus)} pairs)")
    logger.debug(f"Average pairwise Kendall's tau for n={n}: {avg_tau:.4f} (from {len(taus)} pairs)")
    return avg_tau


def assess_narratives(
    model_name: str,
    temperature: float,
    top_p: float,
    dataset: str,
    max_tokens: int,
    repetition_penalty: float,
    max_model_len: int,
    num_narratives: int=20,
) -> None:
    logger.info("=" * 60)
    logger.info(f"Starting narrative assessment for model: {model_name}, dataset: {dataset}")
    logger.info("=" * 60)
    
    set_full_reproducibility()

    LOWER_BOUND = 0
    UPPER_BOUND = 50
    logger.info(f"Processing samples in range [{LOWER_BOUND}, {UPPER_BOUND}]")

    dataset_norm = _normalize_dataset_name(dataset)

    # Prepare output directory & file
    out_dir = os.path.join("results", "number_narratives")
    _ensure_outdir(out_dir)
    out_json = os.path.join(
        out_dir, f"{dataset}_{model_name}_results.json"
    )
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Output JSON file: {out_json}")

    # Load data
    logger.info("Loading counterfactuals data...")
    with open(os.path.join("src", "explainer", "counterfactuals.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded data for {len(data)} datasets")

    # Build prompt base
    base_prompt = prompt

    # Resolve model path or google
    use_google = False
    vllm_llm = None
    tokenizer = None
    sampling_params = None
    google_client = None
    resolved_model = None

    logger.info(f"Resolving model: {model_name}")
    if model_name in MODEL_MAPPING:
        resolved_model = MODEL_MAPPING[model_name]
        logger.info(f"Using vLLM model from MODEL_MAPPING: {resolved_model}")
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(resolved_model)
        logger.info("Creating sampling parameters...")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            top_k=10,
        )
        logger.info(f"Loading vLLM model (this may take a while)...")
        llm_start = time.time()
        vllm_llm = LLM(
            model=resolved_model,
            gpu_memory_utilization=0.92,
            max_model_len=max_model_len,
            max_num_seqs=1,
        )
        llm_time = time.time() - llm_start
        logger.info(f"vLLM model loaded in {llm_time:.2f}s")
    elif model_name in GOOGLE_API_MODEL_MAPPING:
        use_google = True
        resolved_model = GOOGLE_API_MODEL_MAPPING[model_name]
        logger.info(f"Using Google API model from GOOGLE_API_MODEL_MAPPING: {resolved_model}")
        from dotenv import load_dotenv
        from google import genai
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set in environment")
        logger.info("Initializing Google API client...")
        google_client = genai.Client(api_key=api_key)
        logger.info("Google API client initialized successfully")
    else:
        raise ValueError(f"Unknown model '{model_name}'. Not found in MODEL_MAPPING or GOOGLE_API_MODEL_MAPPING.")

    results: Dict[str, Any] = {}
    sample_counter = 0

    for dataset_name, examples in data.items():
        if dataset_name.lower() != dataset_norm:
            continue
        for index, values in examples.items():
            for counterfactual in values["counterfactuals"]:
                if not (LOWER_BOUND <= sample_counter <= UPPER_BOUND):
                    sample_counter += 1
                    continue

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
                features_importance_ranking_lists: List[Dict[str, str]] = []
                for j in range(num_narratives):
                    logger.info(f"  Generating narrative {j+1}/{num_narratives} for sample {sample_counter}...")
                    narr_gen_start = time.time()
                    if not use_google:
                        gen_text = _llm_generate_vllm(
                            vllm_llm, tokenizer, sampling_params, current_prompt, max_retries=3
                        )
                    else:
                        gen_text = _llm_generate_google(
                            google_client, resolved_model, current_prompt, temperature=temperature, top_p=top_p
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
                            if not use_google:
                                gen_text = _llm_generate_vllm(
                                    vllm_llm, tokenizer, sampling_params, current_prompt, max_retries=1
                                )
                            else:
                                gen_text = _llm_generate_google(
                                    google_client, resolved_model, current_prompt, max_retries=1, backoff_seconds=10, temperature=temperature, top_p=top_p
                                )
                            explanation = _extract_explanation_from_text(gen_text)
                            if explanation is not None:
                                retry_ok = True
                                logger.info(f"  Successfully extracted explanation on retry {retry_num+1}")
                                break
                        if not retry_ok:
                            #raise RuntimeError("Failed to extract narrative explanation after 3 attempts")
                            logger.warning(f"  Failed to extract explanation on retry {retry_num+1}, continuing...")
                            continue
                    else:
                        logger.debug(f"  Explanation extracted successfully (length: {len(explanation)} chars)")
                    
                    # Store narrative explanation
                    narratives.append(explanation)

                    # Extract features_importance_ranking (treat missing as None)
                    features_importance_ranking = _extract_features_importance_ranking(gen_text)
                    features_importance_ranking_lists.append(features_importance_ranking if features_importance_ranking else {})
                    num_features = len(features_importance_ranking) if features_importance_ranking else 0
                    logger.info(f"  Narrative {j+1}/{num_narratives} completed (features: {num_features} items)")

                narr_total_time = time.time() - narr_start_time
                logger.info(f"All {num_narratives} narratives generated for sample {sample_counter} in {narr_total_time:.2f}s")

                # Compute per-n average pairwise Jaccard similarities and Kendall's tau
                logger.info(f"Computing pairwise metrics for sample {sample_counter}...")
                sim_comp_start = time.time()
                avg_jaccard: Dict[str, float] = {}
                avg_kendall_tau: Dict[str, float] = {}
                available_narratives = len(features_importance_ranking_lists)
                if available_narratives < num_narratives:
                    logger.info(
                        f"  Only {available_narratives} valid narratives collected (requested {num_narratives}); metrics computed up to this count."
                    )
                for n in range(3, min(num_narratives, available_narratives) + 1):
                    logger.info(f"  Computing metrics for n={n}...")
                    avg_jaccard[str(n)] = _compute_avg_pairwise_jaccard_similarity(features_importance_ranking_lists, n)
                    avg_kendall_tau[str(n)] = _compute_avg_pairwise_kendall_tau(features_importance_ranking_lists, n)
                    logger.info(f"  Average pairwise Jaccard similarity for n={n}: {avg_jaccard[str(n)]:.4f}")
                    logger.info(f"  Average pairwise Kendall's tau for n={n}: {avg_kendall_tau[str(n)]:.4f}")
                sim_comp_time = time.time() - sim_comp_start
                logger.info(f"All metric computations completed for sample {sample_counter} in {sim_comp_time:.2f}s")

                # Store per-sample
                results[str(sample_counter)] = {
                    "dataset": dataset_name,
                    "index": index,
                    "factual": values["factual"],
                    "counterfactual": counterfactual,
                    "narratives": narratives,
                    "features_importance_ranking": features_importance_ranking_lists,
                    "avg_pairwise_jaccard": avg_jaccard,
                    "avg_pairwise_kendall_tau": avg_kendall_tau,
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

    # Aggregate and plot
    logger.info("Aggregating metric data across samples...")
    import matplotlib.pyplot as plt

    counts = list(range(3, num_narratives + 1))
    per_count_jaccard: Dict[int, List[float]] = {c: [] for c in counts}
    per_count_kendall_tau: Dict[int, List[float]] = {c: [] for c in counts}
    
    for sample_id, entry in results.items():
        jaccard_sims = entry.get("avg_pairwise_jaccard", {})
        kendall_tau_sims = entry.get("avg_pairwise_kendall_tau", {})
        for c in counts:
            if str(c) in jaccard_sims and not np.isnan(jaccard_sims[str(c)]):
                per_count_jaccard[c].append(float(jaccard_sims[str(c)]))
            if str(c) in kendall_tau_sims and not np.isnan(kendall_tau_sims[str(c)]):
                per_count_kendall_tau[c].append(float(kendall_tau_sims[str(c)]))

    logger.info("Computing mean and standard deviation for each narrative count...")
    jaccard_means = [np.mean(per_count_jaccard[c]) if per_count_jaccard[c] else np.nan for c in counts]
    jaccard_stds = [np.std(per_count_jaccard[c]) if per_count_jaccard[c] else np.nan for c in counts]
    kendall_means = [np.mean(per_count_kendall_tau[c]) if per_count_kendall_tau[c] else np.nan for c in counts]
    kendall_stds = [np.std(per_count_kendall_tau[c]) if per_count_kendall_tau[c] else np.nan for c in counts]
    
    # Compute mean of the two metrics
    mean_metrics = []
    mean_stds = []
    for idx, c in enumerate(counts):
        j_mean = jaccard_means[idx]
        k_mean = kendall_means[idx]
        if not np.isnan(j_mean) and not np.isnan(k_mean):
            mean_metrics.append((j_mean + k_mean) / 2.0)
            # For std, we approximate by taking the mean of stds (simplified approach)
            mean_stds.append((jaccard_stds[idx] + kendall_stds[idx]) / 2.0)
        else:
            mean_metrics.append(np.nan)
            mean_stds.append(np.nan)
    
    logger.info("Statistics summary:")
    logger.info("Jaccard similarity:")
    for c, mean, std in zip(counts, jaccard_means, jaccard_stds):
        if not np.isnan(mean):
            logger.info(f"  n={c:2d}: mean={mean:.4f}, std={std:.4f}, samples={len(per_count_jaccard[c])}")
    logger.info("Kendall's tau:")
    for c, mean, std in zip(counts, kendall_means, kendall_stds):
        if not np.isnan(mean):
            logger.info(f"  n={c:2d}: mean={mean:.4f}, std={std:.4f}, samples={len(per_count_kendall_tau[c])}")

    # Plot 1: Jaccard similarity
    logger.info("Creating Jaccard similarity plot...")
    plt.figure(figsize=(8, 5))
    plt.errorbar(counts, jaccard_means, yerr=jaccard_stds, fmt='-o', capsize=4)
    plt.xlabel('Number of narratives')
    plt.xticks(counts)
    plt.ylabel('Average pairwise Jaccard similarity')
    plt.title(f'Average pairwise Jaccard similarity vs narratives ({dataset}, {model_name})')
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(out_dir, f"{dataset}_{model_name}_jaccard_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Jaccard plot saved to {plot_path}")
    
    # Plot 2: Kendall's tau
    logger.info("Creating Kendall's tau plot...")
    plt.figure(figsize=(8, 5))
    plt.errorbar(counts, kendall_means, yerr=kendall_stds, fmt='-o', capsize=4)
    plt.xlabel('Number of narratives')
    plt.xticks(counts)
    plt.ylabel('Average pairwise Kendall\'s tau')
    plt.title(f'Average pairwise Kendall\'s tau vs narratives ({dataset}, {model_name})')
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(out_dir, f"{dataset}_{model_name}_kendall_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Kendall's tau plot saved to {plot_path}")
    
    # Plot 3: Mean of both metrics
    logger.info("Creating mean metrics plot...")
    plt.figure(figsize=(8, 5))
    plt.errorbar(counts, mean_metrics, yerr=mean_stds, fmt='-o', capsize=4)
    plt.xlabel('Number of narratives')
    plt.xticks(counts)
    plt.ylabel('Average pairwise similarity (mean of Jaccard & Kendall\'s tau)')
    plt.title(f'Average pairwise similarity (mean) vs narratives ({dataset}, {model_name})')
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(out_dir, f"{dataset}_{model_name}_mean_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Mean plot saved to {plot_path}")
    logger.info("=" * 60)
    logger.info("Narrative assessment completed successfully!")
    logger.info("=" * 60)


__all__ = ["assess_narratives"]


