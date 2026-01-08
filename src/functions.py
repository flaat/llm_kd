import json
import ast
import re
import statistics
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from scipy.stats import kendalltau
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from src.clf import Dataset as DatasetClass
    DATASET_CLASS_AVAILABLE = True
except ImportError:
    DATASET_CLASS_AVAILABLE = False


# ------------------------- Parsing helpers -------------------------

def load_json(file_path: Path) -> Dict:
    """Load JSON data from a file path."""
    with Path(file_path).open("r") as f:
        return json.load(f)


def _try_load_json_snippet(snippet: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse a JSON or Python-literal snippet into a dict."""
    try:
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(snippet)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


def extract_and_parse_json(text: str) -> Optional[Dict]:
    """
    Extract the last JSON-looking object from text and parse it.
    Looks for fenced ```json``` blocks, then any balanced braces.
    Returns a dict or None.
    """
    if not text:
        return None

    candidates: List[str] = []

    # 1) fenced json block
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates.extend([c.strip() for c in fenced if c.strip()])

    # 2) balanced brace scanning
    stack = []
    start_idx = None
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start_idx = i
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start_idx is not None:
                snippet = text[start_idx : i + 1]
                candidates.append(snippet.strip())

    required_keys = {"feature_changes", "reasoning", "features_importance_ranking", "explanation"}
    for cand in reversed(candidates):
        parsed = _try_load_json_snippet(cand)
        if isinstance(parsed, dict) and required_keys.issubset(parsed.keys()):
            return parsed
    for cand in reversed(candidates):
        parsed = _try_load_json_snippet(cand)
        if isinstance(parsed, dict) and "feature_changes" in parsed:
            return parsed
    for cand in reversed(candidates):
        parsed = _try_load_json_snippet(cand)
        if isinstance(parsed, dict):
            return parsed
    return None


def merge_feature_changes(dict_list: Any) -> Dict[str, Dict]:
    """Merge list/dict of feature_changes into a single dict and keep reasonings."""
    result: Dict[str, Dict] = {}
    reasonings: Dict[str, Any] = {}

    if isinstance(dict_list, dict):
        dict_list = [dict_list]
    if not isinstance(dict_list, list):
        return {"feature_changes": result, "reasonings": reasonings}

    for d in dict_list:
        if not isinstance(d, dict):
            continue
        for key, value in d.items():
            if key == "reasoning":
                continue
            result[key] = value
            if "reasoning" in d:
                reasonings[key] = d["reasoning"]
    return {"feature_changes": result, "reasonings": reasonings}


def compute_feature_changes_from_prompt(prompt_text: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Extract factual/counterfactual blocks from a prompt and build feature changes.
    Returns (feature_changes, target_var_change) or (None, None) if parsing fails.
    """
    if not prompt_text:
        return None, None
    try:
        pattern = r"###\s*Factual Example\s*###\s*(\{.*?\})\s*###\s*Counterfactual Example\s*###\s*(\{.*?\})"
        m = re.search(pattern, prompt_text, re.DOTALL)
        if m:
            factual_str, counterfactual_str = m.group(1), m.group(2)
        else:
            dicts = re.findall(r"(\{.*?\})", prompt_text, re.DOTALL)
            if len(dicts) >= 2:
                factual_str, counterfactual_str = dicts[-2], dicts[-1]
            else:
                return None, None

        factual = ast.literal_eval(factual_str)
        counterfactual = ast.literal_eval(counterfactual_str)

        target = None
        # Check for target variables (case-insensitive)
        for key in factual.keys():
            if _is_target_variable(key):
                target = key
                break
        if target is None:
            differing = [k for k in factual.keys() if k in counterfactual and factual[k] != counterfactual[k]]
            if len(differing) == 1:
                target = differing[0]
            elif len(differing) > 1:
                for k in differing:
                    if isinstance(factual[k], int) or isinstance(counterfactual[k], int):
                        target = k
                        break
                if target is None and differing:
                    target = differing[-1]

        feature_changes: Dict[str, Dict[str, Any]] = {}
        for k in factual.keys():
            if k == target:
                continue
            if k in counterfactual and factual[k] != counterfactual[k]:
                feature_changes[k] = {"factual": factual[k], "counterfactual": counterfactual[k]}

        target_var_change = None
        if target and target in factual and target in counterfactual:
            target_var_change = {"factual": factual[target], "counterfactual": counterfactual[target]}

        return feature_changes, target_var_change
    except Exception:
        return None, None


# ------------------------- Metrics -------------------------

def _is_target_variable(var_name: str) -> bool:
    """Check if a variable name is a target variable (case-insensitive)."""
    var_lower = var_name.lower()
    target_vars = ("income", "survived", "target", "medhouseval")
    return var_lower in target_vars


def _normalize_feature_changes(raw_fc: Any) -> Dict[str, Dict]:
    """Normalize feature_changes structure into a dict keyed by feature name."""
    if isinstance(raw_fc, dict):
        return raw_fc
    if isinstance(raw_fc, list):
        normalized: Dict[str, Dict] = {}
        for item in raw_fc:
            if isinstance(item, dict) and len(item) == 1:
                key = next(iter(item.keys()))
                normalized[key] = item[key]
        return normalized
    return {}


def compute_entry_metrics(entry: Dict, parsed: Optional[Dict]) -> Tuple[bool, Optional[bool], Optional[float], Optional[bool]]:
    """
    Compute per-entry metrics.
    Returns (parsed_ok, perfect_feature_match, average_ff, target_correct).
    """
    if parsed is None:
        return False, None, None, None

    # Fill missing ground-truth changes if needed
    if not entry.get("changes"):
        entry["changes"] = {}
        prompt_text = entry.get("prompt") or parsed.get("prompt") or entry.get("generated_text")
        fc, tv = compute_feature_changes_from_prompt(prompt_text)
        if fc:
            entry["changes"]["feature_changes"] = fc
        if tv:
            entry["changes"]["target_variable_change"] = tv

    parsed_fc = parsed.get("feature_changes")
    if not isinstance(parsed_fc, (list, dict)):
        return True, None, None, None

    merged = merge_feature_changes(parsed_fc)
    changes = entry.get("changes", {})
    if not isinstance(changes, dict) or "feature_changes" not in changes:
        return True, None, None, None

    fc_dict = _normalize_feature_changes(changes.get("feature_changes"))
    if not fc_dict:
        return True, None, None, None

    # Separate target variables from feature changes
    # Handle case where target variable might be in feature_changes (validation files)
    # or in target_variable_change (test files)
    target_var_change_gt = changes.get("target_variable_change")
    feature_changes_gt: Dict[str, Dict] = {}
    target_var_gt: Optional[Dict[str, Any]] = None
    
    for var_name, var_data in fc_dict.items():
        if _is_target_variable(var_name):
            # Extract target variable from feature_changes if present
            if target_var_change_gt is None:
                target_var_gt = var_data
        else:
            feature_changes_gt[var_name] = var_data
    
    # If target_variable_change exists separately, use it
    if target_var_change_gt:
        target_var_gt = target_var_change_gt
    
    # Filter parsed feature_changes to exclude target variables
    parsed_feature_changes: Dict[str, Dict] = {}
    for var_name, var_data in merged["feature_changes"].items():
        if not _is_target_variable(var_name):
            parsed_feature_changes[var_name] = var_data

    features_counter = 0
    denom = len(feature_changes_gt)
    target_correct = False
    
    # If number of features doesn't match, perfect_match will be False
    # but we still compute avg_ff based on partial matches
    length_matches = len(parsed_feature_changes) == len(feature_changes_gt)

    # Check target variable correctness
    if target_var_gt is not None:
        try:
            parsed_tv = parsed.get("target_variable_change", {})
            if not parsed_tv and target_var_gt:
                # Try to find target variable in parsed feature_changes (case-insensitive)
                for var_name, var_data in merged["feature_changes"].items():
                    if _is_target_variable(var_name):
                        parsed_tv = var_data
                        break
            
            if parsed_tv:
                factual_gt = target_var_gt.get("factual")
                counterfactual_gt = target_var_gt.get("counterfactual")
                parsed_factual = parsed_tv.get("factual")
                parsed_counterfactual = parsed_tv.get("counterfactual")
                
                # Handle both int and string comparisons
                check_factual = factual_gt == parsed_factual or str(factual_gt) == str(parsed_factual)
                check_counterfactual = counterfactual_gt == parsed_counterfactual or str(counterfactual_gt) == str(parsed_counterfactual)
                target_correct = check_factual and check_counterfactual
        except Exception:
            target_correct = False

    # Check feature changes (excluding target variables)
    for variable, element in feature_changes_gt.items():
        factual_gt = element.get("factual")
        counterfactual_gt = element.get("counterfactual")
        
        try:
            parsed_feat = parsed_feature_changes.get(variable)
            if parsed_feat is None:
                # Try case-insensitive match
                var_lower = variable.lower()
                for parsed_var, parsed_data in parsed_feature_changes.items():
                    if parsed_var.lower() == var_lower:
                        parsed_feat = parsed_data
                        break
            
            if parsed_feat:
                check_factual = factual_gt == parsed_feat.get("factual")
                check_counterfactual = counterfactual_gt == parsed_feat.get("counterfactual")
                if check_factual and check_counterfactual:
                    features_counter += 1
        except Exception:
            pass

    avg_ff = features_counter / denom if denom else None
    # Perfect match only if all features match AND the number of features is correct
    is_perfect = (features_counter == denom and length_matches) if denom else None

    return True, is_perfect, avg_ff, target_correct


def compute_metrics_for_dataset(data: Dict, max_examples: int = 200, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate metrics across a dataset JSON object."""
    parsed_success = 0
    parsed_total = 0
    perfect_ff = 0
    avg_ff_values: List[float] = []
    target_correct_total = 0
    comparable_total = 0  # Number of entries where feature comparison is possible (both avg_ff and perfect_match not None)
    
    # FRA metrics collection
    fra_shap_0_05_values: List[float] = []
    fra_shap_0_1_values: List[float] = []
    fra_shap_0_2_values: List[float] = []
    fra_lime_0_05_values: List[float] = []
    fra_lime_0_1_values: List[float] = []
    fra_lime_0_2_values: List[float] = []
    fra_total_samples = 0

    for idx, key in enumerate(sorted(data.keys(), key=lambda x: int(x))):
        if idx >= max_examples:
            break
        entry = data[key]
        parsed = extract_and_parse_json(entry.get("generated_text", ""))
        parsed_total += 1
        parsed_ok, perfect_match, avg_ff, target_correct = compute_entry_metrics(entry, parsed)

        if parsed_ok:
            parsed_success += 1
        
        # Only count entries where both perfect_match and avg_ff are not None (they should be consistent)
        if perfect_match is not None and avg_ff is not None:
            comparable_total += 1
            avg_ff_values.append(avg_ff)
            if perfect_match:
                perfect_ff += 1
            if target_correct is True:
                target_correct_total += 1
        
        # Compute FRA metrics if conditions are met
        if dataset_name and perfect_match is True and parsed is not None:
            fra_metrics = compute_fra_metrics(entry, parsed, dataset_name)
            
            # Check if any FRA metric was computed (indicating valid analysis)
            if any(v is not None for v in fra_metrics.values()):
                fra_total_samples += 1
                
                if fra_metrics.get("fra_shap_0.05") is not None:
                    fra_shap_0_05_values.append(fra_metrics["fra_shap_0.05"])
                if fra_metrics.get("fra_shap_0.1") is not None:
                    fra_shap_0_1_values.append(fra_metrics["fra_shap_0.1"])
                if fra_metrics.get("fra_shap_0.2") is not None:
                    fra_shap_0_2_values.append(fra_metrics["fra_shap_0.2"])
                if fra_metrics.get("fra_lime_0.05") is not None:
                    fra_lime_0_05_values.append(fra_metrics["fra_lime_0.05"])
                if fra_metrics.get("fra_lime_0.1") is not None:
                    fra_lime_0_1_values.append(fra_metrics["fra_lime_0.1"])
                if fra_metrics.get("fra_lime_0.2") is not None:
                    fra_lime_0_2_values.append(fra_metrics["fra_lime_0.2"])

    parsing_rate = parsed_success / parsed_total if parsed_total else 0.0
    perfect_ff_rate = perfect_ff / comparable_total if comparable_total else 0.0
    avg_ff_rate = statistics.mean(avg_ff_values) if avg_ff_values else 0.0
    avg_ff_std = statistics.stdev(avg_ff_values) if len(avg_ff_values) > 1 else 0.0
    target_f_rate = target_correct_total / comparable_total if comparable_total else 0.0
    
    # Compute FRA averages
    fra_shap_0_05_avg = statistics.mean(fra_shap_0_05_values) if fra_shap_0_05_values else None
    fra_shap_0_1_avg = statistics.mean(fra_shap_0_1_values) if fra_shap_0_1_values else None
    fra_shap_0_2_avg = statistics.mean(fra_shap_0_2_values) if fra_shap_0_2_values else None
    fra_lime_0_05_avg = statistics.mean(fra_lime_0_05_values) if fra_lime_0_05_values else None
    fra_lime_0_1_avg = statistics.mean(fra_lime_0_1_values) if fra_lime_0_1_values else None
    fra_lime_0_2_avg = statistics.mean(fra_lime_0_2_values) if fra_lime_0_2_values else None

    result = {
        "parsing_rate": round(parsing_rate, 4),
        "perfect_ff": round(perfect_ff_rate, 4),
        "avg_ff": round(avg_ff_rate, 4),
        "avg_ff_std": round(avg_ff_std, 4),
        "target_f": round(target_f_rate, 4),
        "fra_shap_0.05": round(fra_shap_0_05_avg, 4) if fra_shap_0_05_avg is not None else None,
        "fra_shap_0.1": round(fra_shap_0_1_avg, 4) if fra_shap_0_1_avg is not None else None,
        "fra_shap_0.2": round(fra_shap_0_2_avg, 4) if fra_shap_0_2_avg is not None else None,
        "fra_lime_0.05": round(fra_lime_0_05_avg, 4) if fra_lime_0_05_avg is not None else None,
        "fra_lime_0.1": round(fra_lime_0_1_avg, 4) if fra_lime_0_1_avg is not None else None,
        "fra_lime_0.2": round(fra_lime_0_2_avg, 4) if fra_lime_0_2_avg is not None else None,
        "fra_total_samples": fra_total_samples,
        "parsed_total": parsed_total,
        "comparable_total": comparable_total,
    }
    
    return result


# ------------------------- Output helpers -------------------------

def build_input_path(dataset_name: str, worker_model: str, refiner: bool, worker_finetuned: bool, refiner_model: Optional[str] = None, refiner_finetuned: bool = False) -> Path:
    if refiner:
        if not refiner_model:
            raise ValueError("refiner_model is required when refiner=True")
        filename = f"{worker_model}--{refiner_model}_response_finetuned_{worker_finetuned}-{refiner_finetuned}.json"
        return Path("results/with_refiner") / dataset_name / f"{worker_model}--{refiner_model}" / filename
    filename = f"{worker_model}_response_finetuned_{worker_finetuned}.json"
    return Path("results/draft_generator") / dataset_name / worker_model / filename


def build_output_dir(refiner: bool) -> Path:
    if refiner:
        return Path("results/with_refiner")
    return Path("results/draft_generator")


def generate_result_row(dataset_name: str, worker_model: str, refiner: bool, metrics: Dict[str, Any], worker_finetuned: bool, refiner_model: Optional[str] = None, refiner_finetuned: bool = False) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "worker_model": worker_model,
        "refiner_model": refiner_model if refiner else None,
        "refiner": refiner,
        "worker_finetuned": worker_finetuned,
        "refiner_finetuned": refiner_finetuned if refiner else None,
        **metrics,
    }


def generate_latex_table(rows: List[Dict[str, Any]], refiner: bool) -> str:
    """Generate a LaTeX table string for the given rows."""
    header = (
        "\\begin{table*}[htbp]\n"
        "    \\centering\n"
        "    \\begin{tabular}{l|l|p{2cm}|p{2cm}|p{2cm}|p{2cm}|p{2cm}}\n"
        "        \\textbf{Worker Model} & \\textbf{Refiner Model} & \\textbf{Avg. FF} $\\uparrow$ & \\textbf{Perfect FF} $\\uparrow$ & \\textbf{TargetF} $\\uparrow$ & \\textbf{Parsing Rate} $\\uparrow$ & \\textbf{FRA} $\\uparrow$ \\\\ \n"
        "        \\midrule \n"
    )
    if not refiner:
        header = (
            "\\begin{table*}[htbp]\n"
            "    \\centering\n"
            "    \\begin{tabular}{l|p{2cm}|p{2cm}|p{2cm}|p{2cm}|p{2cm}}\n"
            "        \\textbf{Worker Model} & \\textbf{Avg. FF} $\\uparrow$ & \\textbf{Perfect FF} $\\uparrow$ & \\textbf{TargetF} $\\uparrow$ & \\textbf{Parsing Rate} $\\uparrow$ & \\textbf{FRA} $\\uparrow$ \\\\ \n"
            "        \\midrule \n"
        )

    body_lines = []
    for row in rows:
        base_cols = [
            row.get("worker_model", "---"),
        ]
        if refiner:
            base_cols.append(row.get("refiner_model", "---"))
        base_cols.extend(
            [
                str(row.get("avg_ff", "n.d.")),
                str(row.get("perfect_ff", "n.d.")),
                str(row.get("target_f", "n.d.")),
                str(row.get("parsing_rate", "n.d.")),
                str(row.get("fra", "n.d.")),
            ]
        )
        body_lines.append(" & ".join(base_cols) + " \\\\")

    footer = (
        "    \\end{tabular} \n"
        "    \\caption{Comparison of results for "
        + ("worker+refiner" if refiner else "worker only")
        + ".}\n"
        "    \\label{tab:results-comparison}\n"
        " \\end{table*}"
    )
    return header + "\n".join(body_lines) + "\n" + footer


def export_results(rows: List[Dict[str, Any]], refiner: bool, output_dir: Path, json_name: str, tex_name: str) -> None:
    """Save JSON results; LaTeX generation is temporarily disabled."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / json_name

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    # NOTE: LaTeX table generation is paused for now. To re-enable, uncomment
    # the lines below.
    # tex_path = output_dir / tex_name
    # table = generate_latex_table(rows, refiner=refiner)
    # with tex_path.open("w") as f:
    #     f.write(table)


# ------------------------- Validation helpers -------------------------

def extract_checkpoint_name(path: Path) -> str:
    stem = path.stem
    last_part = stem.split("_")[-1]
    if last_part.isdigit():
        return last_part
    m = re.search(r"checkpoint-?(\d+)", stem)
    if m:
        return m.group(1)
    return stem


def compute_checkpoint_metrics(file_path: Path, max_examples: int = 200, dataset_name: Optional[str] = None) -> Dict[str, float]:
    data = load_json(file_path)
    metrics = compute_metrics_for_dataset(data, max_examples=max_examples, dataset_name=dataset_name)
    return {
        "parsing_rate": metrics["parsing_rate"],
        "perfect_ff_rate": metrics["perfect_ff"],
        "average_ff": metrics["avg_ff"],
        "parsed_total": metrics["parsed_total"],
        "comparable_total": metrics["comparable_total"],
    }


def collect_validation_metrics(base_dir: Path, datasets: List[str], models: List[str], max_examples: int) -> List[Dict]:
    rows = []
    for dataset in datasets:
        for model in models:
            model_dir = base_dir / dataset / model
            if not model_dir.exists():
                continue
            for json_file in sorted(model_dir.glob("*.json")):
                metrics = compute_checkpoint_metrics(json_file, max_examples=max_examples, dataset_name=dataset)
                checkpoint = extract_checkpoint_name(json_file)
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "checkpoint": checkpoint,
                    **metrics,
                })
    return rows


def save_csv(rows: List[Dict], out_path: Path, fieldnames: List[str]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in fieldnames) + "\n")


def plot_metrics(rows: List[Dict], out_path: Path):
    # Import locally to avoid mandatory matplotlib dependency unless needed.
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    series = defaultdict(list)
    for row in rows:
        key = f"{row['dataset']}::{row['model']}"
        series[key].append(row)

    for key, points in series.items():
        points_sorted = sorted(points, key=lambda r: int(re.sub(r"\\D", "", r["checkpoint"]) or 0))
        xs = [p["checkpoint"] for p in points_sorted]
        axes[0].plot(xs, [p.get("perfect_ff_rate", 0) for p in points_sorted], marker="o", label=key)
        axes[1].plot(xs, [p.get("parsing_rate", 0) for p in points_sorted], marker="o", label=key)
        axes[2].plot(xs, [p.get("average_ff", 0) for p in points_sorted], marker="o", label=key)

    axes[0].set_ylabel("Perfect Feature Match")
    axes[1].set_ylabel("Parsing Success")
    axes[2].set_ylabel("Average FF")
    axes[2].set_xlabel("Checkpoint")
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------- FRA metric helpers -------------------------

# Cache for loaded models and datasets to avoid reloading for each entry
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}

def _normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset name to match pickle file naming convention."""
    # Map common variations to file names
    name_mapping = {
        "Adult Income": "adult",
        "adult income": "adult",
        "adult": "adult",
        "Titanic": "titanic",
        "titanic": "titanic",
        "Diabetes": "diabetes",
        "diabetes": "diabetes",
        "California Housing": "california",
        "california housing": "california",
        "california": "california",
    }
    return name_mapping.get(dataset_name, dataset_name.lower().replace(" ", "_"))


def _load_classifier_model(dataset_name: str) -> Tuple[Any, Any]:
    """
    Load the pickle model and Dataset class for encoding.
    Returns (model, dataset_instance) or (None, None) if loading fails.
    Uses caching to avoid reloading for each entry.
    """
    if not DATASET_CLASS_AVAILABLE:
        return None, None
    
    # Check cache first
    normalized_name = _normalize_dataset_name(dataset_name)
    if normalized_name in _MODEL_CACHE:
        return _MODEL_CACHE[normalized_name]
    
    try:
        model_path = Path("src/explainer/clf_models") / f"{normalized_name}.pkl"
        
        if not model_path.exists():
            _MODEL_CACHE[normalized_name] = (None, None)
            return None, None
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load Dataset class to get encoders and feature order
        # Map normalized name back to Dataset class expected name
        dataset_class_name_mapping = {
            "adult": "Adult Income",
            "titanic": "Titanic",
            "diabetes": "Diabetes",
            "california": "California Housing",
        }
        dataset_class_name = dataset_class_name_mapping.get(normalized_name, dataset_name)
        
        dataset_instance = DatasetClass(name=dataset_class_name)
        
        # Cache the result
        _MODEL_CACHE[normalized_name] = (model, dataset_instance)
        
        return model, dataset_instance
    except Exception:
        _MODEL_CACHE[normalized_name] = (None, None)
        return None, None


def _convert_instance_to_model_input(instance: Dict, dataset: Any) -> Optional[np.ndarray]:
    """
    Convert factual/counterfactual dict to model input format.
    Returns numpy array in correct feature order, or None if conversion fails.
    """
    if dataset is None:
        return None
    
    try:
        # Get feature names in correct order (excluding target)
        feature_names = list(dataset.X_train.columns)
        
        # Build array in correct order
        feature_values = []
        for feat_name in feature_names:
            # Try exact match first
            if feat_name in instance:
                value = instance[feat_name]
            else:
                # Try fuzzy matching: case-insensitive + normalize underscores/hyphens
                value = None
                feat_name_normalized = feat_name.lower().replace('-', '_').replace(' ', '_')
                
                for key in instance.keys():
                    key_normalized = key.lower().replace('-', '_').replace(' ', '_')
                    if key_normalized == feat_name_normalized:
                        value = instance[key]
                        break
                
                # Special mappings for common name variations
                if value is None:
                    # Handle sex/gender synonym
                    if feat_name.lower() in ['sex', 'gender']:
                        for key in instance.keys():
                            if key.lower() in ['sex', 'gender']:
                                value = instance[key]
                                break
                
                if value is None:
                    return None
            
            # Apply label encoder if needed
            if feat_name in dataset.label_encoders:
                le = dataset.label_encoders[feat_name]
                # Convert value to encoded form
                if isinstance(value, str):
                    value = le.transform([value])[0]
                else:
                    # Try to find the encoded value
                    try:
                        value = le.transform([str(value)])[0]
                    except:
                        # If transform fails, try inverse then transform
                        try:
                            decoded = le.inverse_transform([int(value)])[0] if isinstance(value, (int, float)) else value
                            value = le.transform([str(decoded)])[0]
                        except:
                            return None
            
            feature_values.append(value)
        
        return np.array(feature_values, dtype=float)
    except Exception:
        return None


def _compute_shap_importance(
    model, 
    factual_array: np.ndarray, 
    counterfactual_array: np.ndarray, 
    feature_names: List[str], 
    changed_features: List[str]
) -> Optional[Dict[str, float]]:
    """
    Compute SHAP-based importance by calculating |SHAP(x) - SHAP(x')|.
    Returns dict mapping changed feature names to importance magnitudes.
    """
    if not SHAP_AVAILABLE or model is None:
        return None
    
    try:
        # Convert numpy arrays to pandas DataFrames with feature names
        # This avoids the sklearn warning about missing feature names
        factual_df = pd.DataFrame(factual_array.reshape(1, -1), columns=feature_names)
        counterfactual_df = pd.DataFrame(counterfactual_array.reshape(1, -1), columns=feature_names)
        
        # Create SHAP explainer (TreeExplainer for DecisionTree)
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values for both instances using DataFrames
        shap_factual = explainer.shap_values(factual_df)
        shap_counterfactual = explainer.shap_values(counterfactual_df)
        
        # Handle multi-class output: select class based on factual prediction
        if isinstance(shap_factual, list):
            factual_pred = int(model.predict(factual_df)[0])
            shap_factual = shap_factual[factual_pred]
            shap_counterfactual = shap_counterfactual[factual_pred]

        
        # Ensure we have 1D arrays for comparison
        if shap_factual.ndim == 2:
            shap_factual = shap_factual[0]
        if shap_counterfactual.ndim == 2:
            shap_counterfactual = shap_counterfactual[0]
        
        # Ensure same length
        min_len = min(len(shap_factual), len(shap_counterfactual))
        shap_factual = shap_factual[:min_len]
        shap_counterfactual = shap_counterfactual[:min_len]
        
        # Calculate absolute difference (result should be a 1D array)
        shap_diff = np.abs(shap_factual - shap_counterfactual)
        # Ensure shap_diff is a flat 1D numpy array of floats to avoid nested-array issues
        shap_diff = np.asarray(shap_diff, dtype=float).ravel()
        
        # Map to feature names and filter to changed features only
        importance_dict = {}
        changed_feature_lower = [cf.lower() for cf in changed_features]
        
        for idx, feat_name in enumerate(feature_names):
            if idx >= len(shap_diff):
                break
            if feat_name.lower() in changed_feature_lower:
                # Find matching changed feature (case-insensitive)
                matching_feat = None
                for cf in changed_features:
                    if cf.lower() == feat_name.lower():
                        matching_feat = cf
                        break
                if matching_feat:
                    # Extract scalar value safely from shap_diff
                    value = float(shap_diff[idx])
                    importance_dict[matching_feat] = value
        
        return importance_dict if importance_dict else None
    except Exception as e:
        # Print error for debugging
        import traceback
        print(f"SHAP computation error: {e}")
        traceback.print_exc()
        return None


def _compute_lime_importance(
    model,
    factual_array: np.ndarray,
    counterfactual_array: np.ndarray,
    feature_names: List[str],
    changed_features: List[str],
    training_data: np.ndarray
) -> Optional[Dict[str, float]]:
    """
    Compute LIME-based importance using magnitude of weights.
    Returns dict mapping changed feature names to importance magnitudes.
    """
    if not LIME_AVAILABLE or model is None:
        return None
    
    try:
        import warnings
        
        # Create a wrapper function for predict_proba that converts numpy arrays to DataFrames
        # This prevents sklearn warnings about missing feature names
        def predict_proba_wrapper(X):
            """Wrapper that converts numpy arrays to DataFrames before prediction."""
            try:
                if isinstance(X, np.ndarray):
                    # Convert to DataFrame with feature names
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    X_df = pd.DataFrame(X, columns=feature_names)
                    # Suppress the specific warning about feature names
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
                        return model.predict_proba(X_df)
                elif isinstance(X, list):
                    # Convert list to DataFrame
                    X_arr = np.array(X)
                    if X_arr.ndim == 1:
                        X_arr = X_arr.reshape(1, -1)
                    X_df = pd.DataFrame(X_arr, columns=feature_names)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
                        return model.predict_proba(X_df)
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
                        return model.predict_proba(X)
            except Exception as e:
                # Fallback: try without DataFrame conversion if conversion fails
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
                        return model.predict_proba(X)
                except Exception:
                    # Last resort: return without warning suppression
                    return model.predict_proba(X)
        
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            mode='classification',
            discretize_continuous=False
        )
        
        # LIME expects a numpy array (1D), not a list
        # Ensure factual_array is a 1D numpy array
        if isinstance(factual_array, np.ndarray):
            factual_for_lime = factual_array.flatten() if factual_array.ndim > 1 else factual_array
        else:
            factual_for_lime = np.array(factual_array).flatten()
        
        # Get explanation for factual instance using wrapped predict_proba
        explanation = explainer.explain_instance(
            factual_for_lime,
            predict_proba_wrapper,
            num_features=len(feature_names)
        )
        
        # Get predicted class for factual instance
        # Use argmax of predicted probabilities to determine the class index
        factual_proba = predict_proba_wrapper(factual_for_lime.reshape(1, -1))[0]
        class_idx = int(np.argmax(factual_proba))
        
        # Use as_map() to get (feature_index, weight) tuples directly.
        # Keys of as_map() can be class *labels* (model.classes_) rather than
        # plain indices, so we resolve the correct key robustly.
        as_map_dict = explanation.as_map()
        target_key = None
        
        # 1) Direct hit with index key
        if class_idx in as_map_dict:
            target_key = class_idx
        else:
            # 2) Try mapping via model.classes_ if available
            try:
                if hasattr(model, "classes_"):
                    class_label = model.classes_[class_idx]
                    if class_label in as_map_dict:
                        target_key = class_label
            except Exception:
                target_key = None
        
        # 3) Fallback: use first available key to avoid KeyError
        if target_key is None:
            # If there is only one class in as_map, take it
            if len(as_map_dict) == 1:
                target_key = next(iter(as_map_dict.keys()))
            else:
                # Try to select by position; otherwise just take the first key
                try:
                    keys_list = list(as_map_dict.keys())
                    target_key = keys_list[class_idx] if class_idx < len(keys_list) else keys_list[0]
                except Exception:
                    target_key = next(iter(as_map_dict.keys()))
        
        weights = dict(as_map_dict[target_key])  # {feat_idx: weight}
        
        # Build importance dict by mapping feature indices to feature names
        importance_dict = {}
        changed_set = {cf.lower(): cf for cf in changed_features}
        
        for feat_idx, weight in weights.items():
            if feat_idx < len(feature_names):
                feat_name = feature_names[feat_idx]
                # Match using case-insensitive lookup
                matching_feat = changed_set.get(feat_name.lower())
                if matching_feat is not None:
                    importance_dict[matching_feat] = abs(float(weight))
        
        return importance_dict if importance_dict else None
    except Exception as e:
        # Print error for debugging
        import traceback
        print(f"LIME computation error: {e}")
        traceback.print_exc()
        return None


def _compute_tie_factor(magnitudes: List[float], num_changes: int, alpha: float) -> float:
    """
    Compute tie factor based on number of changes.
    If num_changes <= 3: alpha * (max - min)
    If num_changes >= 4: alpha * IQR
    """
    if not magnitudes or len(magnitudes) == 0:
        return 0.0
    
    if num_changes <= 3:
        return alpha * (max(magnitudes) - min(magnitudes))
    else:
        # Compute IQR using numpy percentile
        magnitudes_array = np.asarray(magnitudes)
        if len(magnitudes_array) < 2:
            return 0.0
        q1, q3 = np.percentile(magnitudes_array, [25, 75])
        iqr_value = q3 - q1
        return alpha * iqr_value


def _create_ground_truth_ranking(importance_dict: Dict[str, float], tie_factor: float) -> Dict[str, int]:
    """
    Create ranking with ties based on importance magnitudes.
    Features within tie_factor are assigned the same rank (lower number).
    Returns dict mapping feature names to ranks (1-indexed).
    
    Format matches LLM output: {"feature_x": 1, "feature_y": 2, ...}
    where lower rank number = higher importance.
    
    Uses dense ranks and compares against the anchor of the current tie group
    to avoid incorrect chaining (A tied to B, B tied to C, but A not tied to C).
    """
    if not importance_dict:
        return {}
    
    # Sort features by importance magnitude (descending)
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    ranking = {}
    
    # Handle first feature
    if not sorted_features:
        return ranking
    
    # Initialize with first feature as anchor
    first_feat, first_mag = sorted_features[0]
    group_anchor_mag = first_mag
    current_rank = 1
    ranking[first_feat] = current_rank
    
    # Process remaining features
    for feat_name, magnitude in sorted_features[1:]:
        # Compare against the anchor of the current tie group, not just the previous item
        # This prevents incorrect chaining (A tied to B, B tied to C, but A not tied to C)
        if abs(magnitude - group_anchor_mag) <= tie_factor:
            # Within tie_factor of the group anchor: same rank (tied)
            ranking[feat_name] = current_rank
        else:
            # Outside tie_factor: start new group with dense rank increment
            current_rank += 1
            group_anchor_mag = magnitude  # New anchor for this group
            ranking[feat_name] = current_rank
    
    return ranking


def _compute_kendall_tau_with_ties(ranking1: Dict[str, int], ranking2: Dict[str, int], required_features: Optional[List[str]] = None) -> Optional[float]:
    """
    Compute Kendall tau between two rankings, accounting for ties.
    Returns normalized value: (tau + 1) / 2, or None if computation fails.
    
    Args:
        ranking1: First ranking (ground truth)
        ranking2: Second ranking (predicted)
        required_features: If provided, both rankings must contain all these features.
                          If coverage is incomplete, returns None to avoid inflated scores.
    """
    if not ranking1 or not ranking2:
        return None
    
    # Require full coverage if required_features is provided
    if required_features is not None:
        # Create case-insensitive lookup sets
        ranking1_features_lower = {k.lower(): k for k in ranking1.keys()}
        ranking2_features_lower = {k.lower(): k for k in ranking2.keys()}
        required_set_lower = {rf.lower() for rf in required_features}
        
        # Check if both rankings contain all required features (case-insensitive)
        ranking1_has_all = required_set_lower.issubset(ranking1_features_lower.keys())
        ranking2_has_all = required_set_lower.issubset(ranking2_features_lower.keys())
        
        if not (ranking1_has_all and ranking2_has_all):
            # Incomplete coverage: return None to avoid inflated scores
            return None
    
    # Get common features
    common_features = set(ranking1.keys()) & set(ranking2.keys())
    if len(common_features) < 2:
        return None
    
    # Build lists of ranks for common features
    ranks1 = []
    ranks2 = []
    for feat in sorted(common_features):
        ranks1.append(ranking1[feat])
        ranks2.append(ranking2[feat])
    
    if SCIPY_AVAILABLE:
        try:
            tau, _ = kendalltau(ranks1, ranks2)
            if np.isnan(tau):
                return None
            # Normalize to [0, 1]
            return (tau + 1) / 2
        except Exception:
            return None
    else:
        # Manual implementation (simplified, doesn't handle all tie cases perfectly)
        try:
            n = len(ranks1)
            concordant = 0
            discordant = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    sign1 = np.sign(ranks1[i] - ranks1[j])
                    sign2 = np.sign(ranks2[i] - ranks2[j])
                    if sign1 * sign2 > 0:
                        concordant += 1
                    elif sign1 * sign2 < 0:
                        discordant += 1
            
            total = concordant + discordant
            if total == 0:
                return None
            
            tau = (concordant - discordant) / total
            return (tau + 1) / 2
        except Exception:
            return None


def compute_fra_metrics(entry: Dict, parsed: Dict, dataset_name: str) -> Dict[str, Optional[float]]:
    """
    Compute FRA metrics for an entry.
    Only computes if entry has perfectFF and num_changes > 1.
    Returns dict with 6 FRA values: fra_shap_0.05, fra_shap_0.1, fra_shap_0.2, fra_lime_0.05, fra_lime_0.1, fra_lime_0.2
    """
    result = {
        "fra_shap_0.05": None,
        "fra_shap_0.1": None,
        "fra_shap_0.2": None,
        "fra_lime_0.05": None,
        "fra_lime_0.1": None,
        "fra_lime_0.2": None,
    }
    
    # Check if we have features_importance_ranking in parsed response
    if not parsed or "features_importance_ranking" not in parsed:
        return result
    
    # Get feature changes from entry
    # feature_changes can be either a dict or a list of dicts
    changes = entry.get("changes", {})
    if not isinstance(changes, dict):
        return result
    
    feature_changes_raw = changes.get("feature_changes", {})
    
    # Normalize feature_changes to a dict
    feature_changes = _normalize_feature_changes(feature_changes_raw)
    
    if not feature_changes:
        return result
    
    # Filter out target variables
    feature_changes_filtered = {}
    for var_name, var_data in feature_changes.items():
        if not _is_target_variable(var_name):
            feature_changes_filtered[var_name] = var_data
    
    # Check if num_changes > 1
    if len(feature_changes_filtered) <= 1:
        return result
    
    # Extract factual and counterfactual instances
    # First try to get from ground_truth structure (preferred)
    factual, counterfactual = None, None
    
    ground_truth = entry.get("ground_truth", {})
    if isinstance(ground_truth, dict):
        factual = ground_truth.get("factual")
        counterfactual = ground_truth.get("counterfactual")
    
    # Fall back to extracting from prompt if ground_truth not available
    if not factual or not counterfactual:
        prompt_text = entry.get("prompt") or entry.get("generated_text", "")
        try:
            pattern = r"###\s*Factual Example\s*###\s*(\{.*?\})\s*###\s*Counterfactual Example\s*###\s*(\{.*?\})"
            m = re.search(pattern, prompt_text, re.DOTALL)
            if m:
                factual_str, counterfactual_str = m.group(1), m.group(2)
                factual = ast.literal_eval(factual_str)
                counterfactual = ast.literal_eval(counterfactual_str)
            else:
                dicts = re.findall(r"(\{.*?\})", prompt_text, re.DOTALL)
                if len(dicts) >= 2:
                    factual_str, counterfactual_str = dicts[-2], dicts[-1]
                    factual = ast.literal_eval(factual_str)
                    counterfactual = ast.literal_eval(counterfactual_str)
        except Exception:
            pass
    
    if not factual or not counterfactual:
        return result
    
    # Load model and dataset
    model, dataset = _load_classifier_model(dataset_name)
    if model is None or dataset is None:
        return result
    
    # Convert instances to model input format
    factual_array = _convert_instance_to_model_input(factual, dataset)
    counterfactual_array = _convert_instance_to_model_input(counterfactual, dataset)
    
    if factual_array is None or counterfactual_array is None:
        return result
    
    # Get feature names and changed features
    feature_names = list(dataset.X_train.columns)
    changed_features = list(feature_changes_filtered.keys())
    
    # Get predicted ranking from LLM output
    # Format: {"feature_x": 1, "feature_y": 2, ...} where lower rank = higher importance
    predicted_ranking_raw = parsed.get("features_importance_ranking", {})
    if not isinstance(predicted_ranking_raw, dict):
        return result
    
    # Normalize predicted ranking (convert to int, handle case-insensitive)
    # The LLM produces ranks where 1 = most important, 2 = second most important, etc.
    predicted_ranking = {}
    for feat_name, rank_value in predicted_ranking_raw.items():
        try:
            rank = int(rank_value) if isinstance(rank_value, (int, str)) else None
            if rank is not None:
                # Strip angle brackets if present (models sometimes include them from prompt template)
                clean_feat_name = feat_name.strip('<>') if isinstance(feat_name, str) else feat_name
                
                # Find matching feature in changed_features (case-insensitive)
                matching_feat = None
                for cf in changed_features:
                    if cf.lower() == clean_feat_name.lower():
                        matching_feat = cf
                        break
                if matching_feat:
                    predicted_ranking[matching_feat] = rank
        except (ValueError, TypeError):
            continue
    
    if not predicted_ranking:
        return result
    
    # Get training data for LIME
    training_data = dataset.X_train.values
    
    # Compute for each explainer and alpha combination
    for explainer_name, explainer_func in [("shap", _compute_shap_importance), ("lime", _compute_lime_importance)]:
        for alpha in [0.05, 0.1, 0.2]:
            try:
                # Compute importance
                if explainer_name == "shap":
                    importance_dict = explainer_func(
                        model, factual_array, counterfactual_array, feature_names, changed_features
                    )
                else:  # lime
                    importance_dict = explainer_func(
                        model, factual_array, counterfactual_array, feature_names, changed_features, training_data
                    )
                
                if not importance_dict:
                    continue
                
                # Compute tie factor
                magnitudes = list(importance_dict.values())
                tie_factor = _compute_tie_factor(magnitudes, len(importance_dict), alpha)
                
                # Create ground-truth ranking
                ground_truth_ranking = _create_ground_truth_ranking(importance_dict, tie_factor)
                
                if not ground_truth_ranking:
                    continue
                
                # Compute Kendall tau with full coverage requirement
                # This prevents inflated scores when LLM ranking omits features
                tau_normalized = _compute_kendall_tau_with_ties(ground_truth_ranking, predicted_ranking, required_features=changed_features)
                
                if tau_normalized is not None:
                    key = f"fra_{explainer_name}_{alpha}"
                    result[key] = tau_normalized
            except Exception:
                continue
    
    return result


# ------------------------- Global results helpers -------------------------

def collect_global_results(dataset_name: str, refiner: bool) -> Dict[str, Dict[str, Any]]:
    """
    Scan model directories and collect plain/finetuned results.
    Returns: {model_name: {"plain": {...metrics...}, "ft": {...metrics...}}}
    """
    base_dir = build_output_dir(refiner=refiner)
    dataset_dir = base_dir / dataset_name
    
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    
    results: Dict[str, Dict[str, Any]] = {}
    
    # Scan all model directories
    for model_dir in sorted(dataset_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Look for plain and ft result files
        plain_file = model_dir / f"results_{model_name}-plain.json"
        ft_file = model_dir / f"results_{model_name}-ft.json"
        
        plain_data = None
        ft_data = None
        
        # Load plain results
        if plain_file.exists():
            try:
                plain_json = load_json(plain_file)
                if isinstance(plain_json, list) and len(plain_json) > 0:
                    plain_data = plain_json[0]  # Get first row
            except Exception as e:
                print(f"Warning: Could not load {plain_file}: {e}")
        
        # Load finetuned results
        if ft_file.exists():
            try:
                ft_json = load_json(ft_file)
                if isinstance(ft_json, list) and len(ft_json) > 0:
                    ft_data = ft_json[0]  # Get first row
            except Exception as e:
                print(f"Warning: Could not load {ft_file}: {e}")
        
        # Only add if at least one result exists
        if plain_data is not None or ft_data is not None:
            results[model_name] = {
                "plain": plain_data,
                "ft": ft_data,
            }
    
    return results


def _clean_model_name(model_name: str) -> str:
    """Remove 'unsloth_' prefix from model name for display."""
    if model_name.startswith("unsloth_"):
        return model_name[len("unsloth_"):]
    return model_name


def _clean_model_label(model_name: str) -> str:
	"""Remove unsloth_ prefix for plotting labels."""
	if model_name.startswith("unsloth_"):
		return model_name[len("unsloth_"):]
	return model_name


def _parse_tex_value_pair(cell: str) -> Optional[float]:
	"""Extract the fine-tuned numeric value from a 'plain / ft' LaTeX cell."""
	if not cell:
		return None
	parts = cell.split("/")
	if len(parts) < 2:
		return None
	ft_part = parts[-1]
	# Strip LaTeX row ending and whitespace
	ft_part = ft_part.replace("\\\\", "").strip()
	# If the part contains ±, take the mean before ±
	if "±" in ft_part:
		ft_part = ft_part.split("±")[0].strip()
	try:
		return float(ft_part)
	except ValueError:
		try:
			# Sometimes numbers may carry commas
			return float(ft_part.replace(",", ""))
		except Exception:
			return None


def _parse_global_results_tex(tex_path: Path) -> Dict[str, Dict[str, float]]:
	"""Parse global_results LaTeX table to pull FT PFF and JPR per model."""
	if not tex_path.exists():
		return {}

	results: Dict[str, Dict[str, float]] = {}
	with tex_path.open("r") as f:
		for line in f:
			# Expect rows like: model & avgff & pff & tf & jpr \\
			if "&" not in line or "\\midrule" in line or "textbf" in line:
				continue
			cells = [c.strip() for c in line.split("&")]
			if len(cells) < 5:
				continue
			model = cells[0]
			pff_cell = cells[2]
			jpr_cell = cells[4]
			pff_ft = _parse_tex_value_pair(pff_cell)
			jpr_ft = _parse_tex_value_pair(jpr_cell)
			results[model] = {
				"perfect_ff": pff_ft if pff_ft is not None else None,
				"parsing_rate": jpr_ft if jpr_ft is not None else None,
			}
	return results


def _parse_global_feasibility_tex(tex_path: Path) -> Dict[str, Dict[str, float]]:
	"""Parse feasibility LaTeX table to pull FT inference time mean per model."""
	if not tex_path.exists():
		return {}

	results: Dict[str, Dict[str, float]] = {}
	with tex_path.open("r") as f:
		for line in f:
			if "&" not in line or "\\midrule" in line or "textbf" in line:
				continue
			cells = [c.strip() for c in line.split("&")]
			if len(cells) < 2:
				continue
			model = cells[0]
			inference_cell = cells[1]
			inference_ft = _parse_tex_value_pair(inference_cell)
			results[model] = {"inference_time": inference_ft if inference_ft is not None else None}
	return results


def _format_metric_value(value: Any) -> str:
    """Format metric value for display, handling None and numeric values."""
    if value is None:
        return "n.d."
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _get_metric_pair(plain_data: Dict[str, Any], ft_data: Dict[str, Any], metric_key: str) -> str:
    """Get formatted metric pair: plain_value / ft_value"""
    plain_val = plain_data.get(metric_key) if plain_data else None
    ft_val = ft_data.get(metric_key) if ft_data else None
    
    plain_str = _format_metric_value(plain_val)
    ft_str = _format_metric_value(ft_val)
    
    if plain_str == "n.d." and ft_str == "n.d.":
        return "n.d."
    elif plain_str == "n.d.":
        return ft_str
    elif ft_str == "n.d.":
        return plain_str
    else:
        return f"{plain_str} / {ft_str}"


def _extract_feasibility_stats(data: Dict[str, Any]) -> Dict[str, Any]:
	"""Pull mean/std values from feasibility stats blocks."""
	def _pair(section: str) -> Tuple[Optional[float], Optional[float]]:
		stats = data.get(section, {})
		if isinstance(stats, dict):
			return stats.get("mean"), stats.get("std")
		return None, None
	
	inference_mean, inference_std = _pair("inference_time_stats")
	energy_mean, energy_std = _pair("energy_consumption_stats")
	power_mean, power_std = _pair("average_power_stats")
	
	return {
		"inference_time_mean": inference_mean,
		"inference_time_std": inference_std,
		"energy_consumption_mean": energy_mean,
		"energy_consumption_std": energy_std,
		"average_power_mean": power_mean,
		"average_power_std": power_std,
	}


def collect_global_feasibility_results(dataset_name: str, refiner: bool) -> Dict[str, Dict[str, Any]]:
	"""
	Scan model directories and collect feasibility stats (mean/std) for plain vs fine-tuned runs.
	Returns: {model_name: {"plain": {...}, "ft": {...}}}
	"""
	base_dir = build_output_dir(refiner=refiner)
	dataset_dir = base_dir / dataset_name
	
	if not dataset_dir.exists():
		raise ValueError(f"Dataset directory not found: {dataset_dir}")
	
	results: Dict[str, Dict[str, Any]] = {}
	for model_dir in sorted(dataset_dir.iterdir()):
		if not model_dir.is_dir():
			continue
		
		model_name = model_dir.name
		feas_dir = model_dir / "feasibility"
		if not feas_dir.exists():
			continue
		
		plain_file = feas_dir / f"{model_name}_feasibility_response_finetuned_False.json"
		ft_file = feas_dir / f"{model_name}_feasibility_response_finetuned_True.json"
		
		plain_stats = None
		ft_stats = None
		
		if plain_file.exists():
			try:
				plain_stats = _extract_feasibility_stats(load_json(plain_file))
			except Exception as e:
				print(f"Warning: Could not load feasibility file {plain_file}: {e}")
		if ft_file.exists():
			try:
				ft_stats = _extract_feasibility_stats(load_json(ft_file))
			except Exception as e:
				print(f"Warning: Could not load feasibility file {ft_file}: {e}")
		
		if plain_stats is not None or ft_stats is not None:
			results[model_name] = {"plain": plain_stats, "ft": ft_stats}
	
	return results


def collect_overall_metrics(datasets: List[str], refiner: bool) -> Dict[str, Dict[str, float]]:
	"""
	Aggregate fine-tuned metrics across datasets.
	Returns: {model: {"parsing_rate": avg, "perfect_ff": avg, "inference_time": avg}}
	"""
	aggregates: Dict[str, Dict[str, List[float]]] = {}

	def _append(model: str, key: str, value: Any) -> None:
		if value is None:
			return
		aggregates.setdefault(model, {}).setdefault(key, []).append(float(value))

	for dataset in datasets:
		base_dir = build_output_dir(refiner=refiner) / "global_results"
		suffix = "with_refiner" if refiner else "draft_generator"
		main_tex = base_dir / f"{dataset}_{suffix}.tex"
		feas_tex = base_dir / f"{dataset}_{suffix}_feasibility.tex"

		main_results = _parse_global_results_tex(main_tex)
		feas_results = _parse_global_feasibility_tex(feas_tex)

		if not main_results and not feas_results:
			print(f"Warning: No global_results tables found for dataset '{dataset}'")
			continue

		for model_name, vals in main_results.items():
			_append(model_name, "perfect_ff", vals.get("perfect_ff"))
			_append(model_name, "parsing_rate", vals.get("parsing_rate"))

		for model_name, vals in feas_results.items():
			_append(model_name, "inference_time", vals.get("inference_time"))

	overall: Dict[str, Dict[str, float]] = {}
	for model_name, metric_lists in aggregates.items():
		averaged: Dict[str, float] = {}
		for metric_key, values in metric_lists.items():
			if values:
				averaged[metric_key] = sum(values) / len(values)
		if averaged:
			overall[model_name] = averaged

	return overall


def plot_overall_metrics(overall_metrics: Dict[str, Dict[str, float]], datasets: List[str], refiner: bool, output_path: Path) -> None:
	import matplotlib.pyplot as plt
	import numpy as np

	models = sorted(overall_metrics.keys())
	if not models:
		print("Warning: No overall metrics to plot.")
		return

	clean_model_names = [_clean_model_label(m) for m in models]
	metrics = [
		("parsing_rate", "JPR"),
		("perfect_ff", "PFF"),
		("inference_time", "Inference Time (s)"),
	]

	fig, axes = plt.subplots(1, 3, figsize=(15, 4))
	colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
	x = np.arange(len(models))

	for idx, (metric_key, metric_label) in enumerate(metrics):
		ax = axes[idx]
		values = [float(overall_metrics[m].get(metric_key, 0.0)) for m in models]
		ax.bar(x, values, color=colors, alpha=0.85)
		ax.set_title(metric_label, fontsize=11, fontweight="bold")
		ax.set_xticks(x)
		ax.set_xticklabels(clean_model_names, rotation=45, ha="right", fontsize=9)
		ax.grid(True, linestyle="--", alpha=0.3, axis="y")
		ax.set_ylabel(metric_label, fontsize=10)

	mode_str = "Worker+Refiner" if refiner else "Worker Only"
	plt.suptitle(f"Overall ({', '.join(datasets)}) - {mode_str}", fontsize=14, fontweight="bold")
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def _format_mean_std(mean: Any, std: Any) -> str:
	"""Format mean±std string, handling None."""
	if mean is None:
		return "n.d."
	mean_val = f"{float(mean):.3f}" if isinstance(mean, (int, float)) else str(mean)
	if std is None:
		return mean_val
	std_val = f"{float(std):.3f}" if isinstance(std, (int, float)) else str(std)
	return f"{mean_val} ± {std_val}"


def _get_mean_std_pair(plain_data: Dict[str, Any], ft_data: Dict[str, Any], base_key: str) -> str:
	"""Format plain/ft pair for mean/std metrics."""
	plain_mean = plain_data.get(f"{base_key}_mean") if plain_data else None
	plain_std = plain_data.get(f"{base_key}_std") if plain_data else None
	ft_mean = ft_data.get(f"{base_key}_mean") if ft_data else None
	ft_std = ft_data.get(f"{base_key}_std") if ft_data else None
	
	plain_str = _format_mean_std(plain_mean, plain_std)
	ft_str = _format_mean_std(ft_mean, ft_std)
	
	if plain_str == "n.d." and ft_str == "n.d.":
		return "n.d."
	if plain_str == "n.d.":
		return ft_str
	if ft_str == "n.d.":
		return plain_str
	return f"{plain_str} / {ft_str}"


def generate_global_latex_table(results: Dict[str, Dict[str, Any]], dataset_name: str, refiner: bool) -> str:
    """Generate LaTeX table with plain/finetuned values in same column separated by '/' (no FRA column)."""
    header = (
        "\\begin{table*}[htbp]\n"
        "    \\centering\n"
        "    \\begin{tabular}{l|p{2.5cm}|p{2.5cm}|p{2.5cm}|p{2.5cm}}\n"
        "        \\textbf{Model} & \\textbf{Avg. FF} \\newline (Plain / FT) & \\textbf{PFF} \\newline (Plain / FT) & \\textbf{TF} \\newline (Plain / FT) & \\textbf{JPR} \\newline (Plain / FT) \\\\ \n"
        "        \\midrule \n"
    )
    
    body_lines = []
    for model_name in sorted(results.keys()):
        model_data = results[model_name]
        plain_data = model_data.get("plain")
        ft_data = model_data.get("ft")
        
        clean_name = _clean_model_name(model_name)
        
        # Format each metric as plain / ft (order: Avg. FF, PFF, TF, JPR)
        avg_ff = _get_metric_pair(plain_data, ft_data, "avg_ff")
        perfect_ff = _get_metric_pair(plain_data, ft_data, "perfect_ff")
        target_f = _get_metric_pair(plain_data, ft_data, "target_f")
        parsing_rate = _get_metric_pair(plain_data, ft_data, "parsing_rate")
        
        # Order: Model, Avg. FF, PFF, TF, JPR
        row = " & ".join([clean_name, avg_ff, perfect_ff, target_f, parsing_rate]) + " \\\\"
        body_lines.append(row)
    
    # Build footer without f-string issues
    mode_str = "worker+refiner" if refiner else "worker only"
    label_suffix = "-refiner" if refiner else "-draft"
    footer = (
        "    \\end{tabular} \n"
        "    \\caption{Comparison of results for " + mode_str + " on " + dataset_name + " dataset.}\n"
        "    \\label{tab:results-" + dataset_name + label_suffix + "}\n"
        " \\end{table*}"
    )
    
    return header + "\n".join(body_lines) + "\n" + footer


def generate_global_barplots(results: Dict[str, Dict[str, Any]], dataset_name: str, refiner: bool, output_path: Path) -> None:
    """Generate barplots with 4 subplots (2x2) showing metrics comparison (no FRA plot)."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Prepare data
    models = sorted(results.keys())
    clean_model_names = [_clean_model_name(m) for m in models]
    
    # Only 4 metrics here; FRA has its own dedicated plots.
    metrics = [
        ("avg_ff", "Avg. FF"),
        ("perfect_ff", "Perfect FF"),
        ("target_f", "TargetF"),
        ("parsing_rate", "Parsing Rate"),
    ]
    
    # Create figure with 2 rows and 2 columns (two plots per row)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Get color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    x = np.arange(len(models))
    width = 0.35  # Width of bars
    
    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx]
        
        plain_values: List[float] = []
        ft_values: List[float] = []
        
        for model_name in models:
            model_data = results[model_name]
            plain_data = model_data.get("plain")
            ft_data = model_data.get("ft")
            
            plain_val = plain_data.get(metric_key) if plain_data else None
            ft_val = ft_data.get(metric_key) if ft_data else None
            
            # Treat None as 0.0 for plotting
            plain_values.append(float(plain_val) if plain_val is not None else 0.0)
            ft_values.append(float(ft_val) if ft_val is not None else 0.0)
        
        # Create all plain bars first, then all finetuned bars
        ax.bar(
            x - width / 2,
            plain_values,
            width,
            color=colors,
            alpha=0.8,
            label="Plain" if idx == 0 else "",
        )
        ax.bar(
            x + width / 2,
            ft_values,
            width,
            color=colors,
            alpha=0.8,
            hatch="///",
            edgecolor="black",
            linewidth=0.5,
            label="Fine-Tuned" if idx == 0 else "",
        )
        
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(clean_model_names, rotation=45, ha="right", fontsize=9)
        max_val = max(max(plain_values), max(ft_values)) if plain_values or ft_values else 0.0
        ax.set_ylim(0, max_val * 1.1 if max_val > 0 else 1.0)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    
    # Add legend only on first subplot
    if len(models) > 0:
        axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    
    plt.suptitle(f"{dataset_name} - {'Worker+Refiner' if refiner else 'Worker Only'}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_global_feasibility_latex_table(results: Dict[str, Dict[str, Any]], dataset_name: str, refiner: bool) -> str:
	"""Generate LaTeX table for feasibility metrics (mean±std) showing plain/ft pairs."""
	header = (
		"\\begin{table*}[htbp]\n"
		"    \\centering\n"
		"    \\begin{tabular}{l|p{3cm}|p{3cm}|p{3cm}}\n"
		"        \\textbf{Model} & \\textbf{Inference Time (s)} \\\\ (Plain / FT) & "
		"\\textbf{Energy (J)} \\\\ (Plain / FT) & "
		"\\textbf{Avg. Power (W)} \\\\ (Plain / FT) \\\\ \n"
		"        \\midrule \n"
	)
	
	body_lines: List[str] = []
	for model_name in sorted(results.keys()):
		model_data = results[model_name]
		plain_data = model_data.get("plain")
		ft_data = model_data.get("ft")
		
		clean_name = _clean_model_name(model_name)
		
		inference_pair = _get_mean_std_pair(plain_data, ft_data, "inference_time")
		energy_pair = _get_mean_std_pair(plain_data, ft_data, "energy_consumption")
		power_pair = _get_mean_std_pair(plain_data, ft_data, "average_power")
		
		row = " & ".join([clean_name, inference_pair, energy_pair, power_pair]) + " \\\\"
		body_lines.append(row)
	
	mode_str = "worker+refiner" if refiner else "worker only"
	label_suffix = "-refiner-feas" if refiner else "-draft-feas"
	footer = (
		"    \\end{tabular} \n"
		"    \\caption{Feasibility metrics (mean $\\pm$ std) for " + mode_str + " on " + dataset_name + " dataset.}\n"
		"    \\label{tab:feasibility-results-" + dataset_name + label_suffix + "}\n"
		" \\end{table*}"
	)
	
	return header + "\n".join(body_lines) + "\n" + footer


def generate_global_feasibility_barplots(
	results: Dict[str, Dict[str, Any]],
	dataset_name: str,
	refiner: bool,
	output_path: Path,
) -> None:
	"""Generate barplots for feasibility metrics with error bars showing std."""
	import matplotlib.pyplot as plt
	import numpy as np
	
	models = sorted(results.keys())
	if not models:
		return
	
	clean_model_names = [_clean_model_name(m) for m in models]
	metrics = [
		("inference_time", "Inference Time (s)"),
		("energy_consumption", "Energy (J)"),
		("average_power", "Average Power (W)"),
	]
	
	fig, axes = plt.subplots(1, 3, figsize=(16, 5))
	colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
	x = np.arange(len(models))
	width = 0.35
	
	for idx, (metric_key, metric_label) in enumerate(metrics):
		ax = axes[idx]
		
		plain_means: List[float] = []
		ft_means: List[float] = []
		plain_stds: List[float] = []
		ft_stds: List[float] = []
		
		for model_name in models:
			model_data = results[model_name]
			plain_data = model_data.get("plain") or {}
			ft_data = model_data.get("ft") or {}
			
			plain_mean = plain_data.get(f"{metric_key}_mean")
			plain_std = plain_data.get(f"{metric_key}_std")
			ft_mean = ft_data.get(f"{metric_key}_mean")
			ft_std = ft_data.get(f"{metric_key}_std")
			
			plain_means.append(float(plain_mean) if plain_mean is not None else 0.0)
			ft_means.append(float(ft_mean) if ft_mean is not None else 0.0)
			plain_stds.append(float(plain_std) if plain_std is not None else 0.0)
			ft_stds.append(float(ft_std) if ft_std is not None else 0.0)
		
		ax.bar(
			x - width / 2,
			plain_means,
			width,
			color=colors,
			alpha=0.85,
			yerr=plain_stds,
			capsize=4,
			label="Plain" if idx == 0 else "",
		)
		ax.bar(
			x + width / 2,
			ft_means,
			width,
			color=colors,
			alpha=0.85,
			hatch="///",
			edgecolor="black",
			linewidth=0.5,
			yerr=ft_stds,
			capsize=4,
			label="Fine-Tuned" if idx == 0 else "",
		)
		
		ax.set_xlabel("Model", fontsize=10)
		ax.set_ylabel(metric_label, fontsize=10)
		ax.set_title(metric_label, fontsize=11, fontweight="bold")
		ax.set_xticks(x)
		ax.set_xticklabels(clean_model_names, rotation=45, ha="right", fontsize=9)
		max_val = max(max(plain_means, default=0.0), max(ft_means, default=0.0))
		ax.set_ylim(0, max_val * 1.1 if max_val > 0 else 1.0)
		ax.grid(True, linestyle="--", alpha=0.3, axis="y")
	
	axes[0].legend(loc="upper left", fontsize=8, ncol=2)
	plt.suptitle(
		f"{dataset_name} - {'Worker+Refiner' if refiner else 'Worker Only'} Feasibility Metrics",
		fontsize=14,
		fontweight="bold",
	)
	plt.tight_layout()
	
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)


def generate_global_fra_latex_table(results: Dict[str, Dict[str, Any]], dataset_name: str, refiner: bool) -> str:
    """
    Generate LaTeX table for FRA metrics only.
    Columns: Model, FRA SHAP 0.05, FRA SHAP 0.1, FRA SHAP 0.2, FRA LIME 0.05, FRA LIME 0.1, FRA LIME 0.2.
    Each FRA cell shows Plain / FT; if a value is None it is rendered as n.d.
    """
    header = (
        "\\begin{table*}[htbp]\n"
        "    \\centering\n"
        "    \\begin{tabular}{l|p{2.8cm}|p{2.8cm}|p{2.8cm}|p{2.8cm}}\n"
        "        \\textbf{Model} & \\textbf{FRA SHAP 0.05} \\\\ (Plain / FT) & "
        "\\textbf{FRA SHAP 0.1} \\\\ (Plain / FT) & "
        "\\textbf{FRA SHAP 0.2} \\\\ (Plain / FT) & "
        "\\textbf{FRA LIME 0.05} \\\\ (Plain / FT) & "
        "\\textbf{FRA LIME 0.1} \\\\ (Plain / FT) & "
        "\\textbf{FRA LIME 0.2} \\\\ (Plain / FT) \\\\ \n"
        "\\textbf{FRA LIME 0.1} \\\\ (Plain / FT) \\\\ \n"
        "        \\midrule \n"
    )

    body_lines: List[str] = []
    for model_name in sorted(results.keys()):
        model_data = results[model_name]
        plain_data = model_data.get("plain")
        ft_data = model_data.get("ft")

        clean_name = _clean_model_name(model_name)

        fra_shap_005 = _get_metric_pair(plain_data, ft_data, "fra_shap_0.05")
        fra_shap_01 = _get_metric_pair(plain_data, ft_data, "fra_shap_0.1")
        fra_shap_02 = _get_metric_pair(plain_data, ft_data, "fra_shap_0.2")
        fra_lime_005 = _get_metric_pair(plain_data, ft_data, "fra_lime_0.05")
        fra_lime_01 = _get_metric_pair(plain_data, ft_data, "fra_lime_0.1")
        fra_lime_02 = _get_metric_pair(plain_data, ft_data, "fra_lime_0.2")
        fra_lime_01 = _get_metric_pair(plain_data, ft_data, "fra_lime_0.1")

        row = " & ".join(
            [clean_name, fra_shap_005, fra_shap_01, fra_shap_02, fra_lime_005, fra_lime_01, fra_lime_02]
        ) + " \\\\"
        body_lines.append(row)

    mode_str = "worker+refiner" if refiner else "worker only"
    label_suffix = "-refiner-fra" if refiner else "-draft-fra"
    footer = (
        "    \\end{tabular} \n"
        "    \\caption{FRA metrics for " + mode_str + " on " + dataset_name + " dataset.}\n"
        "    \\label{tab:fra-results-" + dataset_name + label_suffix + "}\n"
        " \\end{table*}"
    )

    return header + "\n".join(body_lines) + "\n" + footer


def generate_global_fra_barplots(
    results: Dict[str, Dict[str, Any]],
    dataset_name: str,
    refiner: bool,
    output_path: Path,
) -> None:
    """
    Generate barplots for FRA metrics only.
    Layout: 2x3 subplots (three plots per row).
    None values are treated as 0 in the plots.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    models = sorted(results.keys())
    if not models:
        return

    clean_model_names = [_clean_model_name(m) for m in models]

    metrics = [
        ("fra_shap_0.05", "FRA SHAP 0.05"),
        ("fra_shap_0.1", "FRA SHAP 0.1"),
        ("fra_shap_0.2", "FRA SHAP 0.2"),
        ("fra_lime_0.05", "FRA LIME 0.05"),
        ("fra_lime_0.1", "FRA LIME 0.1"),
        ("fra_lime_0.2", "FRA LIME 0.2"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    x = np.arange(len(models))
    width = 0.35

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx]

        plain_values: List[float] = []
        ft_values: List[float] = []

        for model_name in models:
            model_data = results[model_name]
            plain_data = model_data.get("plain")
            ft_data = model_data.get("ft")

            plain_val = plain_data.get(metric_key) if plain_data else None
            ft_val = ft_data.get(metric_key) if ft_data else None

            # None -> 0.0 for plotting
            plain_values.append(float(plain_val) if plain_val is not None else 0.0)
            ft_values.append(float(ft_val) if ft_val is not None else 0.0)

        ax.bar(
            x - width / 2,
            plain_values,
            width,
            color=colors,
            alpha=0.8,
            label="Plain" if idx == 0 else "",
        )
        ax.bar(
            x + width / 2,
            ft_values,
            width,
            color=colors,
            alpha=0.8,
            hatch="///",
            edgecolor="black",
            linewidth=0.5,
            label="Fine-Tuned" if idx == 0 else "",
        )

        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(clean_model_names, rotation=45, ha="right", fontsize=9)

        max_val = max(max(plain_values), max(ft_values)) if plain_values or ft_values else 0.0
        ax.set_ylim(0, max_val * 1.1 if max_val > 0 else 1.0)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    # Legend only on first subplot
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)

    plt.suptitle(
        f"{dataset_name} - {'Worker+Refiner' if refiner else 'Worker Only'} FRA Metrics",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------- Number Narratives NCS helpers -------------------------

def compute_jaccard_similarity(dict1: Dict[str, int], dict2: Dict[str, int]) -> float:
    """
    Compute Jaccard similarity between two dictionaries based on their keys (feature names).
    
    J(A, B) = |A ∩ B| / |A ∪ B|
    
    Returns:
        float: Jaccard similarity in [0, 1]. Two empty dicts have similarity 1.0.
    """
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


def compute_kendall_tau_normalized(dict1: Dict[str, int], dict2: Dict[str, int]) -> float:
    """
    Compute Kendall's tau between two rankings and normalize to [0, 1] as (tau + 1) / 2.
    
    Args:
        dict1: First ranking {feature_name: rank}
        dict2: Second ranking {feature_name: rank}
    
    Returns:
        float: Normalized Kendall's tau in [0, 1], or 0.5 if computation fails.
    """
    if not dict1 or not dict2:
        return 0.5  # Neutral value when computation not possible
    
    # Get common features (case-insensitive matching)
    dict1_lower = {k.lower(): v for k, v in dict1.items()}
    dict2_lower = {k.lower(): v for k, v in dict2.items()}
    common_features = set(dict1_lower.keys()) & set(dict2_lower.keys())
    
    if len(common_features) < 2:
        return 0.5  # Need at least 2 features for correlation
    
    # Build rank lists
    ranks1 = []
    ranks2 = []
    for feat in sorted(common_features):
        ranks1.append(dict1_lower[feat])
        ranks2.append(dict2_lower[feat])
    
    if SCIPY_AVAILABLE:
        try:
            tau, _ = kendalltau(ranks1, ranks2)
            if np.isnan(tau):
                return 0.5
            # Normalize to [0, 1]
            return (tau + 1) / 2
        except Exception:
            return 0.5
    else:
        # Manual implementation
        try:
            n = len(ranks1)
            concordant = 0
            discordant = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    sign1 = np.sign(ranks1[i] - ranks1[j])
                    sign2 = np.sign(ranks2[i] - ranks2[j])
                    if sign1 * sign2 > 0:
                        concordant += 1
                    elif sign1 * sign2 < 0:
                        discordant += 1
            
            total = concordant + discordant
            if total == 0:
                return 0.5
            
            tau = (concordant - discordant) / total
            return (tau + 1) / 2
        except Exception:
            return 0.5


def compute_pairwise_coherence_score(
    dict1: Dict[str, int],
    dict2: Dict[str, int],
    alpha: float = 0.6
) -> float:
    """
    Compute pairwise coherence score S(e_i, e_j) between two feature importance rankings.
    
    S(e_i, e_j) = J(e_i, e_j)^alpha * ((tau(e_i, e_j) + 1) / 2)^(1-alpha)
    
    Where:
    - J: Jaccard similarity on feature names (keys)
    - tau: Kendall's tau correlation, normalized to [0, 1]
    
    Args:
        dict1: First ranking {feature_name: rank}
        dict2: Second ranking {feature_name: rank}
        alpha: Weight for Jaccard similarity (default 0.6)
    
    Returns:
        float: Pairwise coherence score in [0, 1]
    """
    jaccard = compute_jaccard_similarity(dict1, dict2)
    tau_normalized = compute_kendall_tau_normalized(dict1, dict2)
    
    # S = J^alpha * tau_norm^(1-alpha)
    score = (jaccard ** alpha) * (tau_normalized ** (1 - alpha))
    return score


def compute_narrative_coherence_score(
    rankings: List[Dict[str, int]],
    alpha: float = 0.6
) -> float:
    """
    Compute Narrative Coherence Score (NCS) for a set of feature importance rankings.
    
    NCS = (2 / N(N-1)) * sum_{i<j} S(e_i, e_j)
    
    Where S is the pairwise coherence score.
    
    Args:
        rankings: List of ranking dictionaries {feature_name: rank}
        alpha: Weight for Jaccard similarity in pairwise score (default 0.6)
    
    Returns:
        float: NCS in [0, 1], or NaN if not enough rankings
    """
    n = len(rankings)
    if n < 2:
        return float("nan")
    
    # Compute sum of pairwise coherence scores
    total_score = 0.0
    num_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            score = compute_pairwise_coherence_score(rankings[i], rankings[j], alpha)
            total_score += score
            num_pairs += 1
    
    # NCS = (2 / N(N-1)) * sum = sum / num_pairs
    if num_pairs == 0:
        return float("nan")
    
    ncs = total_score / num_pairs
    return ncs


def collect_number_narratives_metrics(
    base_dir: Path,
    datasets: List[str],
    models: List[str],
    num_narratives: int,
    n_start: int = 3,
    alphas: List[float] = [0.6]
) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
    """
    Collect NCS, Jaccard, and Ranking metrics for number-of-narratives analysis.
    
    For each sample, for each N in [n_start, K]:
    - Enumerate ALL C(K, N) subsets of the K narratives
    - Compute NCS for each subset (for each alpha)
    - Compute average pairwise Jaccard similarity
    - Compute average pairwise Ranking (Kendall tau) similarity
    - Aggregate across all samples: mean and std per N per model
    
    Args:
        base_dir: Base directory containing results (e.g., results/number_narratives)
        datasets: List of dataset names to process
        models: List of model names to process
        num_narratives: K value (max narratives generated per sample)
        n_start: Minimum N for subset sampling (default 3)
        alphas: List of alpha parameters for coherence score (default [0.6])
    
    Returns:
        Dict structure: {dataset: {model: {N: {"ncs_alpha_0.5": {...}, "ncs_alpha_0.6": {...}, 
                "ncs_alpha_0.8": {...}, "jaccard": {...}, "ranking": {...}}}}}
    """
    from itertools import combinations
    
    results: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    
    for dataset in datasets:
        results[dataset] = {}
        
        for model in models:
            # Load JSON file for this model/dataset
            json_path = base_dir / dataset / f"{model}.json"
            
            if not json_path.exists():
                print(f"Warning: File not found: {json_path}")
                continue
            
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {json_path}: {e}")
                continue
            
            # Collect metrics per N across all samples
            # Structure: {n: {"ncs_alpha_X": [...], "jaccard": [...], "ranking": [...]}}
            metrics_per_n: Dict[int, Dict[str, List[float]]] = {
                n: {f"ncs_alpha_{alpha}": [] for alpha in alphas}
                for n in range(n_start, num_narratives + 1)
            }
            for n in range(n_start, num_narratives + 1):
                metrics_per_n[n]["jaccard"] = []
                metrics_per_n[n]["ranking"] = []
            
            for sample_id, sample_data in data.items():
                rankings = sample_data.get("features_importance_ranking", [])
                
                # Filter out empty rankings
                valid_rankings = [r for r in rankings if r]
                
                if len(valid_rankings) < n_start:
                    # Not enough valid rankings for this sample
                    continue
                
                # For each N, enumerate all C(K, N) subsets and compute metrics
                k = len(valid_rankings)
                for n in range(n_start, min(num_narratives, k) + 1):
                    # Get all combinations of size n
                    for subset_indices in combinations(range(k), n):
                        subset_rankings = [valid_rankings[i] for i in subset_indices]
                        
                        # Compute NCS for each alpha
                        for alpha in alphas:
                            ncs = compute_narrative_coherence_score(subset_rankings, alpha)
                            if not np.isnan(ncs):
                                metrics_per_n[n][f"ncs_alpha_{alpha}"].append(ncs)
                        
                        # Compute average pairwise Jaccard similarity
                        jaccard_values = []
                        for i in range(len(subset_rankings)):
                            for j in range(i + 1, len(subset_rankings)):
                                jaccard = compute_jaccard_similarity(subset_rankings[i], subset_rankings[j])
                                jaccard_values.append(jaccard)
                        if jaccard_values:
                            avg_jaccard = np.mean(jaccard_values)
                            metrics_per_n[n]["jaccard"].append(avg_jaccard)
                        
                        # Compute average pairwise Ranking (Kendall tau) similarity
                        ranking_values = []
                        for i in range(len(subset_rankings)):
                            for j in range(i + 1, len(subset_rankings)):
                                ranking = compute_kendall_tau_normalized(subset_rankings[i], subset_rankings[j])
                                ranking_values.append(ranking)
                        if ranking_values:
                            avg_ranking = np.mean(ranking_values)
                            metrics_per_n[n]["ranking"].append(avg_ranking)
            
            # Compute mean and std for each N and each metric
            model_results: Dict[int, Dict[str, Any]] = {}
            for n in range(n_start, num_narratives + 1):
                model_results[n] = {}
                
                # Process NCS for each alpha
                for alpha in alphas:
                    key = f"ncs_alpha_{alpha}"
                    values = metrics_per_n[n][key]
                    if values:
                        model_results[n][key] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "count": len(values),
                        }
                    else:
                        model_results[n][key] = {
                            "mean": float("nan"),
                            "std": float("nan"),
                            "count": 0,
                        }
                
                # Process Jaccard
                jaccard_values = metrics_per_n[n]["jaccard"]
                if jaccard_values:
                    model_results[n]["jaccard"] = {
                        "mean": float(np.mean(jaccard_values)),
                        "std": float(np.std(jaccard_values)),
                        "count": len(jaccard_values),
                    }
                else:
                    model_results[n]["jaccard"] = {
                        "mean": float("nan"),
                        "std": float("nan"),
                        "count": 0,
                    }
                
                # Process Ranking
                ranking_values = metrics_per_n[n]["ranking"]
                if ranking_values:
                    model_results[n]["ranking"] = {
                        "mean": float(np.mean(ranking_values)),
                        "std": float(np.std(ranking_values)),
                        "count": len(ranking_values),
                    }
                else:
                    model_results[n]["ranking"] = {
                        "mean": float("nan"),
                        "std": float("nan"),
                        "count": 0,
                    }
            
            results[dataset][model] = model_results
    
    return results


def plot_number_narratives_metrics(
    metrics: Dict[str, Dict[int, Dict[str, Any]]],
    output_path: Path,
    dataset_name: str,
    metric_key: str,
    metric_label: str,
    n_start: int = 3,
    num_narratives: int = 8
) -> None:
    """
    Generate plot of a specific metric (with std error bars) vs N for all models.
    
    Args:
        metrics: Dict structure {model: {N: {metric_key: {"mean": float, "std": float}}}}
        output_path: Path to save the plot
        dataset_name: Name of the dataset for title
        metric_key: Key to extract from metrics (e.g., "ncs_alpha_0.6", "jaccard", "ranking")
        metric_label: Label for y-axis (e.g., "NCS (α=0.6)", "Jaccard Similarity", "Ranking Similarity")
        n_start: Minimum N value (default 3)
        num_narratives: Maximum N value (default 8)
    """
    import matplotlib.pyplot as plt
    
    if not metrics:
        print(f"Warning: No metrics to plot for {dataset_name}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # X-axis values
    n_values = list(range(n_start, num_narratives + 1))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    
    for idx, (model_name, model_metrics) in enumerate(sorted(metrics.items())):
        means = []
        stds = []
        valid_n = []
        
        for n in n_values:
            if n in model_metrics and metric_key in model_metrics[n]:
                metric_data = model_metrics[n][metric_key]
                if isinstance(metric_data, dict) and not np.isnan(metric_data.get("mean", np.nan)):
                    means.append(metric_data["mean"])
                    stds.append(metric_data["std"])
                    valid_n.append(n)
        
        if valid_n:
            # Clean model name for display
            display_name = model_name
            if display_name.startswith("unsloth_"):
                display_name = display_name[len("unsloth_"):]
            
            ax.errorbar(
                valid_n,
                means,
                yerr=stds,
                marker='o',
                capsize=4,
                color=colors[idx],
                label=display_name,
                linewidth=2,
                markersize=6
            )
    
    ax.set_xlabel("Number of Narratives (N)", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"{metric_label} vs Number of Narratives - {dataset_name}", fontsize=14, fontweight="bold")
    ax.set_xticks(n_values)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")

