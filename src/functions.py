import json
import ast
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


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


def compute_metrics_for_dataset(data: Dict, max_examples: int = 200) -> Dict[str, Any]:
    """Aggregate metrics across a dataset JSON object."""
    parsed_success = 0
    parsed_total = 0
    perfect_ff = 0
    avg_ff_values: List[float] = []
    target_correct_total = 0
    comparable_total = 0

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

    parsing_rate = parsed_success / parsed_total if parsed_total else 0.0
    perfect_ff_rate = perfect_ff / parsed_total if parsed_total else 0.0
    avg_ff_rate = statistics.mean(avg_ff_values) if avg_ff_values else 0.0
    avg_ff_std = statistics.stdev(avg_ff_values) if len(avg_ff_values) > 1 else 0.0
    target_f_rate = target_correct_total / parsed_total if parsed_total else 0.0

    return {
        "parsing_rate": round(parsing_rate, 4),
        "perfect_ff": round(perfect_ff_rate, 4),
        "avg_ff": round(avg_ff_rate, 4),
        "avg_ff_std": round(avg_ff_std, 4),
        "target_f": round(target_f_rate, 4),
        "fra": None,  # placeholder until defined
        "parsed_total": parsed_total,
        "comparable_total": comparable_total,
    }


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


def compute_checkpoint_metrics(file_path: Path, max_examples: int = 200) -> Dict[str, float]:
    data = load_json(file_path)
    metrics = compute_metrics_for_dataset(data, max_examples=max_examples)
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
                metrics = compute_checkpoint_metrics(json_file, max_examples=max_examples)
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
