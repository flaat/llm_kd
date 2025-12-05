"""Aggregate validation metrics across checkpoints.

This script extracts metrics from generated JSON outputs produced during fine-tuning
validation runs. It focuses on two metrics per checkpoint: Perfect Feature Match and
Parsed JSON Success Rate. The logic is adapted from `src/evaluate.py` but refactored
to be modular and reusable.
"""

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# ------------------------- Parsing helpers (from evaluate.py) -------------------------

def load_json(file_path: Path):
	"""Load JSON data from a file."""
	with file_path.open("r") as f:
		return json.load(f)


def extract_and_parse_json(text: str, file_label: str):
	"""
	Extract the last JSON object from text and parse it. Looks for objects containing
	feature_changes, reasoning, features_importance_ranking, and explanation.
	Returns dict or None.
	"""
	def try_load(s: str):
		try:
			return json.loads(s)
		except Exception:
			try:
				return ast.literal_eval(s)
			except Exception:
				return None

	# 1) fenced json block
	fence = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
	candidates = [c.strip() for c in fence if c.strip()]

	# 2) balanced brace scanning to gather all brace-delimited substrings
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

	# prefer last candidate that looks like the final JSON with required keys
	required_keys = {"feature_changes", "reasoning", "features_importance_ranking", "explanation"}
	for cand in reversed(candidates):
		parsed = try_load(cand)
		if isinstance(parsed, dict) and required_keys.issubset(parsed.keys()):
			return parsed
	# fallback: any dict with feature_changes
	for cand in reversed(candidates):
		parsed = try_load(cand)
		if isinstance(parsed, dict) and "feature_changes" in parsed:
			return parsed
	for cand in reversed(candidates):
		parsed = try_load(cand)
		if isinstance(parsed, dict):
			return parsed

	return None


def merge_dicts(dict_list: List[Dict]):
	"""Merge list of single-key dicts; skip invalid entries (e.g., ellipsis)."""
	result = {}
	reasonings = {}
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
	Extract factual and counterfactual examples from a prompt and build feature changes.
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
		for t in ("income", "Survived", "survived", "target"):
			if t in factual:
				target = t
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
				if target is None:
					target = differing[-1]

		feature_changes = {}
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


# ------------------------- Metric computations -------------------------

def compute_entry_metrics(entry: Dict, parsed: Dict) -> Tuple[bool, Optional[bool], Optional[float]]:
	"""
	Compute per-entry metrics, matching evaluate.py logic exactly.
	Returns (parsed_ok, perfect_feature_match, average_ff).
	
	Mirrors evaluate.py line 178-202: iterates over ground truth changes["feature_changes"],
	comparing to parsed values. Target variables (income/Survived) compared against
	parsed["target_variable_change"], others against merged["feature_changes"].
	"""

	if parsed is None:
		return False, None, None

	# Build or backfill changes structure
	if not entry.get("changes"):
		entry["changes"] = {}
		prompt_text = entry.get("prompt") or parsed.get("prompt") or entry.get("generated_text")
		fc, tv = compute_feature_changes_from_prompt(prompt_text)
		if fc:
			entry["changes"]["feature_changes"] = fc
		if tv:
			entry["changes"]["target_variable_change"] = tv

	if "feature_changes" not in parsed:
		return True, None, None

	parsed_fc = parsed.get("feature_changes")
	if not isinstance(parsed_fc, (list, dict)):
		return True, None, None

	merged = merge_dicts(parsed_fc)
	changes = entry.get("changes", {})
	if not isinstance(changes, dict) or "feature_changes" not in changes:
		return True, None, None

	# Normalize ground truth feature_changes (list of dicts) into a dict
	raw_fc = changes.get("feature_changes", {})
	if isinstance(raw_fc, list):
		fc_dict = {}
		for item in raw_fc:
			if isinstance(item, dict) and len(item) == 1:
				key = next(iter(item.keys()))
				fc_dict[key] = item[key]
	elif isinstance(raw_fc, dict):
		fc_dict = raw_fc
	else:
		fc_dict = {}

	if not fc_dict:
		return True, None, None

	# Validate parsed has same number of features
	if len(merged["feature_changes"]) != len(fc_dict):
		return True, None, None

	# Iterate ground truth features and compare with parsed values
	# (evaluate.py lines 178-202)
	features_counter = 0
	denom = len(fc_dict)
	
	try:
		for variable, element in fc_dict.items():
			factual_gt = element.get("factual")
			counterfactual_gt = element.get("counterfactual")
			
			if variable in ("income", "Survived"):
				# Target variable: compare against parsed["target_variable_change"]
				try:
					parsed_tv = parsed.get("target_variable_change", {})
					check_factual = factual_gt == int(parsed_tv.get("factual", -999))
					check_counterfactual = counterfactual_gt == int(parsed_tv.get("counterfactual", -999))
					if check_factual and check_counterfactual:
						features_counter += 1
				except (ValueError, TypeError, KeyError):
					pass
			else:
				# Non-target feature: compare against merged feature values
				try:
					parsed_feat = merged["feature_changes"].get(variable, {})
					check_factual = factual_gt == parsed_feat.get("factual")
					check_counterfactual = counterfactual_gt == parsed_feat.get("counterfactual")
					if check_factual and check_counterfactual:
						features_counter += 1
				except (TypeError, KeyError):
					pass
	except Exception:
		return True, None, None

	# Compute metrics matching evaluate.py logic
	avg_ff = features_counter / denom
	is_perfect = features_counter == denom
	
	return True, is_perfect, avg_ff


def compute_checkpoint_metrics(file_path: Path, max_examples: int = 200) -> Dict[str, float]:
	"""Compute metrics for a single checkpoint JSON file."""

	data = load_json(file_path)
	parsed_success = 0
	perfect_ff = 0
	avg_ff_sum = 0.0
	avg_ff_count = 0
	parsed_total = 0
	comparable_total = 0

	for idx, key in enumerate(sorted(data.keys(), key=lambda x: int(x))):
		if idx >= max_examples:
			break
		entry = data[key]
		parsed = extract_and_parse_json(entry.get("generated_text", ""), file_path.name)
		parsed_total += 1
		parsed_ok, perfect_match, avg_ff = compute_entry_metrics(entry, parsed)
		if parsed_ok:
			parsed_success += 1
		if perfect_match is not None:
			comparable_total += 1
			if perfect_match:
				perfect_ff += 1
		if avg_ff is not None:
			avg_ff_sum += avg_ff
			avg_ff_count += 1

	parsing_rate = parsed_success / parsed_total if parsed_total else 0.0
	perfect_ff_rate = perfect_ff / comparable_total if comparable_total else 0.0
	avg_ff_rate = avg_ff_sum / avg_ff_count if avg_ff_count else 0.0
	return {
		"parsing_rate": parsing_rate,
		"perfect_ff_rate": perfect_ff_rate,
		"average_ff": avg_ff_rate,
		"parsed_total": parsed_total,
		"comparable_total": comparable_total,
		"avg_ff_count": avg_ff_count,
	}


# ------------------------- Collection and I/O -------------------------

def collect_metrics(base_dir: Path, datasets: List[str], models: List[str], max_examples: int) -> List[Dict]:
	"""Iterate datasets/models and compute metrics per checkpoint."""

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


def extract_checkpoint_name(path: Path) -> str:
	stem = path.stem
	# Prefer trailing numeric segment after last underscore
	last_part = stem.split("_")[-1]
	if last_part.isdigit():
		return last_part
	# Fallback to checkpoint-123 style
	m = re.search(r"checkpoint-?(\d+)", stem)
	if m:
		return m.group(1)
	return stem


def save_csv(rows: List[Dict], out_path: Path, fieldnames: List[str]):
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_metrics(rows: List[Dict], out_path: Path):
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

	# Group by dataset/model for plotting distinct lines
	series = defaultdict(list)
	for row in rows:
		key = f"{row['dataset']}::{row['model']}"
		series[key].append(row)

	for key, points in series.items():
		points_sorted = sorted(points, key=lambda r: int(re.sub(r"\D", "", r["checkpoint"]) or 0))
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


# ------------------------- CLI -------------------------

def parse_args():
	parser = argparse.ArgumentParser(description="Aggregate validation metrics across checkpoints.")
	parser.add_argument("--validation", action="store_true", help="Use validation results path (worker_validation).")
	parser.add_argument("--datasets", nargs="+", default=["adult"], help="Datasets to include (space-separated list).")
	parser.add_argument("--models", nargs="+",
					default=[
						"unsloth_qwen_0.5B",
                        "unsloth_qwen3_0.6B",
						"unsloth_deepseek_r1_qwen_1.5B",
                        "unsloth_qwen3_1.7B",
                        "unsloth_llama_3B-Instruct",
                        "unsloth_qwen_3B",
						"unsloth_qwen3_4B-Thinking",  
                    ], help="Models to include (space-separated list).")
	parser.add_argument("--max-examples", type=int, default=200, help="Max examples per checkpoint file to evaluate.")
	parser.add_argument("--output-dir", type=Path, default=None, help="Root directory for outputs; defaults to base_dir/<dataset>/<model>.")
	return parser.parse_args()


def main():
	args = parse_args()
	if not args.validation:
		raise SystemExit("Only validation mode is implemented. Pass --validation.")

	base_dir = Path("results/fine-tuning/worker_validation")
	rows = collect_metrics(base_dir, datasets=args.datasets, models=args.models, max_examples=args.max_examples)

	if not rows:
		raise SystemExit("No metrics collected. Check dataset/model names or input files.")

	# Determine root for outputs: default to base_dir so results land under base_dir/<dataset>/<model>
	output_root = args.output_dir if args.output_dir else base_dir

	# Group rows per dataset/model and write inside each model folder
	grouped = defaultdict(list)
	for row in rows:
		grouped[(row["dataset"], row["model"])].append(row)

	for (dataset, model), group_rows in grouped.items():
		out_dir = output_root / dataset / model
		summary_plot = out_dir / "summary.png"
		perfect_csv = out_dir / "perfectFF.csv"
		parsing_csv = out_dir / "parsing.csv"
		avg_csv = out_dir / "averageFF.csv"

		perfect_rows = [r for r in group_rows if "perfect_ff_rate" in r]
		parsing_rows = [r for r in group_rows if "parsing_rate" in r]
		avg_rows = [r for r in group_rows if "average_ff" in r]
		save_csv(perfect_rows, perfect_csv, ["dataset", "model", "checkpoint", "perfect_ff_rate", "comparable_total"])
		save_csv(parsing_rows, parsing_csv, ["dataset", "model", "checkpoint", "parsing_rate", "parsed_total"])
		save_csv(avg_rows, avg_csv, ["dataset", "model", "checkpoint", "average_ff", "avg_ff_count"])

		plot_metrics(group_rows, summary_plot)


if __name__ == "__main__":
	main()
