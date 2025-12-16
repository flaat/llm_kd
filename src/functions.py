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

    # Compare lengths (excluding target variables)
    if len(parsed_feature_changes) != len(feature_changes_gt):
        return True, None, None, None

    features_counter = 0
    denom = len(feature_changes_gt)
    target_correct = False

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
    is_perfect = features_counter == denom if denom else None

    return True, is_perfect, avg_ff, target_correct


def compute_metrics_for_dataset(data: Dict, max_examples: int = 200, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate metrics across a dataset JSON object."""
    parsed_success = 0
    parsed_total = 0
    perfect_ff = 0
    avg_ff_values: List[float] = []
    target_correct_total = 0
    comparable_total = 0
    
    # FRA metrics collection
    fra_shap_0_05_values: List[float] = []
    fra_shap_0_1_values: List[float] = []
    fra_lime_0_05_values: List[float] = []
    fra_lime_0_1_values: List[float] = []
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
        if perfect_match is not None:
            comparable_total += 1
            if perfect_match:
                perfect_ff += 1
        if avg_ff is not None:
            avg_ff_values.append(avg_ff)
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
                if fra_metrics.get("fra_lime_0.05") is not None:
                    fra_lime_0_05_values.append(fra_metrics["fra_lime_0.05"])
                if fra_metrics.get("fra_lime_0.1") is not None:
                    fra_lime_0_1_values.append(fra_metrics["fra_lime_0.1"])

    parsing_rate = parsed_success / parsed_total if parsed_total else 0.0
    perfect_ff_rate = perfect_ff / parsed_total if parsed_total else 0.0
    avg_ff_rate = statistics.mean(avg_ff_values) if avg_ff_values else 0.0
    avg_ff_std = statistics.stdev(avg_ff_values) if len(avg_ff_values) > 1 else 0.0
    target_f_rate = target_correct_total / parsed_total if parsed_total else 0.0
    
    # Compute FRA averages
    fra_shap_0_05_avg = statistics.mean(fra_shap_0_05_values) if fra_shap_0_05_values else None
    fra_shap_0_1_avg = statistics.mean(fra_shap_0_1_values) if fra_shap_0_1_values else None
    fra_lime_0_05_avg = statistics.mean(fra_lime_0_05_values) if fra_lime_0_05_values else None
    fra_lime_0_1_avg = statistics.mean(fra_lime_0_1_values) if fra_lime_0_1_values else None

    result = {
        "parsing_rate": round(parsing_rate, 4),
        "perfect_ff": round(perfect_ff_rate, 4),
        "avg_ff": round(avg_ff_rate, 4),
        "avg_ff_std": round(avg_ff_std, 4),
        "target_f": round(target_f_rate, 4),
        "fra_shap_0.05": round(fra_shap_0_05_avg, 4) if fra_shap_0_05_avg is not None else None,
        "fra_shap_0.1": round(fra_shap_0_1_avg, 4) if fra_shap_0_1_avg is not None else None,
        "fra_lime_0.05": round(fra_lime_0_05_avg, 4) if fra_lime_0_05_avg is not None else None,
        "fra_lime_0.1": round(fra_lime_0_1_avg, 4) if fra_lime_0_1_avg is not None else None,
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
    """
    if not DATASET_CLASS_AVAILABLE:
        return None, None
    
    try:
        # Normalize dataset name
        normalized_name = _normalize_dataset_name(dataset_name)
        model_path = Path("src/explainer/clf_models") / f"{normalized_name}.pkl"
        
        if not model_path.exists():
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
        
        return model, dataset_instance
    except Exception:
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
                # Try case-insensitive match
                value = None
                for key in instance.keys():
                    if key.lower() == feat_name.lower():
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
        
        # Handle multi-class output (take first class if list)
        if isinstance(shap_factual, list):
            shap_factual = shap_factual[0]
        if isinstance(shap_counterfactual, list):
            shap_counterfactual = shap_counterfactual[0]
        
        # Ensure we have 1D arrays for comparison
        # Flatten to 1D if needed
        if shap_factual.ndim > 1:
            shap_factual = shap_factual.flatten()
        if shap_counterfactual.ndim > 1:
            shap_counterfactual = shap_counterfactual.flatten()
        
        # Ensure same length
        min_len = min(len(shap_factual), len(shap_counterfactual))
        shap_factual = shap_factual[:min_len]
        shap_counterfactual = shap_counterfactual[:min_len]
        
        # Calculate absolute difference (result is 1D array)
        shap_diff = np.abs(shap_factual - shap_counterfactual)
        
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
                    # Extract scalar value from numpy array
                    value = shap_diff[idx]
                    if isinstance(value, np.ndarray):
                        value = value.item() if value.size == 1 else float(value[0])
                    else:
                        value = float(value)
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
        
        # Extract feature importance (absolute values)
        importance_dict = {}
        changed_feature_lower = [cf.lower() for cf in changed_features]
        
        # explanation.as_list() returns list of (feature_name, importance) tuples
        for feat_name_str, importance in explanation.as_list():
            # feat_name_str is typically the feature name as string
            # Try to match it to our feature names (case-insensitive)
            matching_feat = None
            for cf in changed_features:
                # Check if the LIME feature name matches our changed feature
                if (cf.lower() in feat_name_str.lower() or 
                    feat_name_str.lower() in cf.lower() or
                    cf.lower() == feat_name_str.lower()):
                    matching_feat = cf
                    break
            
            # If no direct match, try matching by index in feature_names
            if matching_feat is None:
                try:
                    # Sometimes LIME returns feature names with indices like "feature_0"
                    # Try to extract index
                    import re
                    idx_match = re.search(r'(\d+)', feat_name_str)
                    if idx_match:
                        feat_idx = int(idx_match.group(1))
                        if feat_idx < len(feature_names):
                            feat_name = feature_names[feat_idx]
                            for cf in changed_features:
                                if cf.lower() == feat_name.lower():
                                    matching_feat = cf
                                    break
                except:
                    pass
            
            if matching_feat:
                importance_dict[matching_feat] = abs(importance)
        
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
    """
    if not importance_dict:
        return {}
    
    # Sort features by importance magnitude (descending)
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    ranking = {}
    current_rank = 1
    
    for i, (feat_name, magnitude) in enumerate(sorted_features):
        if i == 0:
            # First feature always gets rank 1
            ranking[feat_name] = current_rank
        else:
            # Check if this feature is within tie_factor of any previous feature in the same tied group
            # We check against the most recent feature's magnitude
            prev_magnitude = sorted_features[i - 1][1]
            if abs(magnitude - prev_magnitude) <= tie_factor:
                # Same rank as previous (tied)
                ranking[feat_name] = current_rank
            else:
                # New rank: count how many features we've seen so far (this is the position)
                # Rank should be the position of the first feature in this new group
                current_rank = i + 1
                ranking[feat_name] = current_rank
    
    return ranking


def _compute_kendall_tau_with_ties(ranking1: Dict[str, int], ranking2: Dict[str, int]) -> Optional[float]:
    """
    Compute Kendall tau between two rankings, accounting for ties.
    Returns normalized value: (tau + 1) / 2, or None if computation fails.
    """
    if not ranking1 or not ranking2:
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
    Returns dict with 4 FRA values: fra_shap_0.05, fra_shap_0.1, fra_lime_0.05, fra_lime_0.1
    """
    result = {
        "fra_shap_0.05": None,
        "fra_shap_0.1": None,
        "fra_lime_0.05": None,
        "fra_lime_0.1": None,
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
                # Find matching feature in changed_features (case-insensitive)
                matching_feat = None
                for cf in changed_features:
                    if cf.lower() == feat_name.lower():
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
        for alpha in [0.05, 0.1]:
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
                tie_factor = _compute_tie_factor(magnitudes, len(changed_features), alpha)
                
                # Create ground-truth ranking
                ground_truth_ranking = _create_ground_truth_ranking(importance_dict, tie_factor)
                
                if not ground_truth_ranking:
                    continue
                
                # Compute Kendall tau
                tau_normalized = _compute_kendall_tau_with_ties(ground_truth_ranking, predicted_ranking)
                
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


def generate_global_latex_table(results: Dict[str, Dict[str, Any]], dataset_name: str, refiner: bool) -> str:
    """Generate LaTeX table with plain/finetuned values in same column separated by '/'."""
    header = (
        "\\begin{table*}[htbp]\n"
        "    \\centering\n"
        "    \\begin{tabular}{l|p{2.5cm}|p{2.5cm}|p{2.5cm}|p{2.5cm}|p{2.5cm}}\n"
        "        \\textbf{Model} & \\textbf{Avg. FF} \\newline (Plain / FT) & \\textbf{PFF} \\newline (Plain / FT) & \\textbf{TF} \\newline (Plain / FT) & \\textbf{FRA} \\newline (Plain / FT) & \\textbf{JPR} \\newline (Plain / FT) \\\\ \n"
        "        \\midrule \n"
    )
    
    body_lines = []
    for model_name in sorted(results.keys()):
        model_data = results[model_name]
        plain_data = model_data.get("plain")
        ft_data = model_data.get("ft")
        
        clean_name = _clean_model_name(model_name)
        
        # Format each metric as plain / ft (order: Avg. FF, PFF, TF, FRA, JPR)
        avg_ff = _get_metric_pair(plain_data, ft_data, "avg_ff")
        perfect_ff = _get_metric_pair(plain_data, ft_data, "perfect_ff")
        target_f = _get_metric_pair(plain_data, ft_data, "target_f")
        fra = _get_metric_pair(plain_data, ft_data, "fra")
        parsing_rate = _get_metric_pair(plain_data, ft_data, "parsing_rate")
        
        # Order: Model, Avg. FF, PFF, TF, FRA, JPR
        row = " & ".join([clean_name, avg_ff, perfect_ff, target_f, fra, parsing_rate]) + " \\\\"
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
    """Generate barplots with 5 subplots showing metrics comparison."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Prepare data
    models = sorted(results.keys())
    clean_model_names = [_clean_model_name(m) for m in models]
    
    metrics = [
        ("avg_ff", "Avg. FF"),
        ("perfect_ff", "Perfect FF"),
        ("target_f", "TargetF"),
        ("fra", "FRA"),
        ("parsing_rate", "Parsing Rate"),
    ]
    
    # Create figure with 2 rows: 3 plots on first row, 2 plots on second row
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Get color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    x = np.arange(len(models))
    width = 0.35  # Width of bars
    
    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx]
        
        plain_values = []
        ft_values = []
        
        for model_name in models:
            model_data = results[model_name]
            plain_data = model_data.get("plain")
            ft_data = model_data.get("ft")
            
            plain_val = plain_data.get(metric_key) if plain_data else None
            ft_val = ft_data.get(metric_key) if ft_data else None
            
            plain_values.append(plain_val if plain_val is not None else 0)
            ft_values.append(ft_val if ft_val is not None else 0)
        
        # Create all plain bars first, then all finetuned bars
        plain_bars = ax.bar(x - width/2, plain_values, width, color=colors, alpha=0.8, label="Plain" if idx == 0 else "")
        ft_bars = ax.bar(x + width/2, ft_values, width, color=colors, alpha=0.8, hatch='///', edgecolor='black', linewidth=0.5, label="Fine-Tuned" if idx == 0 else "")
        
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(clean_model_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, max(max(plain_values), max(ft_values)) * 1.1 if max(plain_values) > 0 or max(ft_values) > 0 else 1.0)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Hide the 6th subplot (index 5) since we only have 5 metrics
    axes[5].axis('off')
    
    # Add legend only on first subplot
    if len(models) > 0:
        axes[0].legend(loc='upper left', fontsize=8, ncol=2)
    
    plt.suptitle(f"{dataset_name} - {'Worker+Refiner' if refiner else 'Worker Only'}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
