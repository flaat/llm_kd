"""Generate results via unified evaluation (global) or legacy validation mode."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from src.evaluate import evaluate_single, evaluate_and_save  # type: ignore  # noqa
from src.functions import (  # type: ignore  # noqa
	build_output_dir,
	export_results,
	collect_validation_metrics,
	save_csv,
	plot_metrics,
	collect_global_results,
	generate_global_latex_table,
	generate_global_barplots,
)


# ------------------------- CLI -------------------------

def parse_args():
	parser = argparse.ArgumentParser(description="Generate results (global) or validation metrics.")
	parser.add_argument("--validation", action="store_true", help="Run validation mode on results/fine-tuning/worker_validation.")
	parser.add_argument("--global", action="store_true", dest="global_mode", help="Generate global results by scanning all model directories.")
	parser.add_argument("--datasets", nargs="+", default=["adult"], help="Datasets to include for validation.")
	parser.add_argument("--models", nargs="+", default=[], help="Models to include for validation.")
	parser.add_argument("--max-examples", type=int, default=200, help="Max examples per checkpoint file.")
	parser.add_argument("--output-dir", type=Path, default=None, help="Root directory for validation outputs; defaults to base_dir/<dataset>/<model>.")

	# Single-run (non-validation) convenience flags
	parser.add_argument("--dataset-name", type=str, help="Dataset name for single run (non-validation).")
	parser.add_argument("--worker-model", type=str, help="Worker model name for single run.")
	parser.add_argument("--refiner", action="store_true", help="Enable refiner for single run.")
	parser.add_argument("--worker-finetuned", action="store_true", help="Worker finetuned flag for single run.")
	parser.add_argument("--refiner-model", type=str, help="Refiner model name for single run (required if --refiner).")
	parser.add_argument("--refiner-finetuned", action="store_true", help="Refiner finetuned flag for single run.")
	return parser.parse_args()


def main():
	args = parse_args()

	if args.validation:
		if not args.models:
			raise SystemExit("Provide --models for validation mode.")
		base_dir = Path("results/fine-tuning/worker_validation")
		rows = collect_validation_metrics(base_dir, datasets=args.datasets, models=args.models, max_examples=args.max_examples)
		if not rows:
			raise SystemExit("No metrics collected. Check dataset/model names or input files.")

		output_root = args.output_dir if args.output_dir else base_dir
		grouped = defaultdict(list)
		for row in rows:
			grouped[(row["dataset"], row["model"])].append(row)

		for (dataset, model), group_rows in grouped.items():
			out_dir = output_root / dataset / model
			summary_plot = out_dir / "summary.png"
			perfect_csv = out_dir / "perfectFF.csv"
			parsing_csv = out_dir / "parsing.csv"
			avg_csv = out_dir / "averageFF.csv"

			save_csv(group_rows, perfect_csv, ["dataset", "model", "checkpoint", "perfect_ff_rate", "comparable_total"])
			save_csv(group_rows, parsing_csv, ["dataset", "model", "checkpoint", "parsing_rate", "parsed_total"])
			save_csv(group_rows, avg_csv, ["dataset", "model", "checkpoint", "average_ff", "parsed_total"])
			plot_metrics(group_rows, summary_plot)
		return

	# Global mode (scan all models)
	if args.global_mode:
		if not args.dataset_name:
			raise SystemExit("Provide --dataset-name for global mode.")
		
		try:
			results = collect_global_results(args.dataset_name, args.refiner)
		except ValueError as e:
			raise SystemExit(str(e))
		if not results:
			raise SystemExit(f"No results found for dataset '{args.dataset_name}' in {'with_refiner' if args.refiner else 'draft_generator'} directory.")
		
		# Generate LaTeX table
		latex_table = generate_global_latex_table(results, args.dataset_name, args.refiner)
		
		# Save LaTeX table and PNG to global_results directory
		base_dir = build_output_dir(args.refiner)
		output_dir = base_dir / "global_results"
		suffix = "with_refiner" if args.refiner else "draft_generator"
		tex_filename = f"{args.dataset_name}_{suffix}.tex"
		tex_path = output_dir / tex_filename
		tex_path.parent.mkdir(parents=True, exist_ok=True)
		with tex_path.open("w") as f:
			f.write(latex_table)
		print(f"LaTeX table saved to: {tex_path}")
		
		# Generate and save barplots
		png_filename = f"{args.dataset_name}_{suffix}.png"
		png_path = output_dir / png_filename
		generate_global_barplots(results, args.dataset_name, args.refiner, png_path)
		print(f"Barplots saved to: {png_path}")
		
		return

	# Single-run (non-validation, non-global) mode
	if not args.dataset_name or not args.worker_model:
		raise SystemExit("Provide --dataset-name and --worker-model for a single run.")
	if args.refiner and not args.refiner_model:
		raise SystemExit("Provide --refiner-model when --refiner is set.")

	evaluate_and_save(
		dataset_name=args.dataset_name,
		worker_model=args.worker_model,
		refiner=args.refiner,
		worker_finetuned=args.worker_finetuned,
		refiner_model=args.refiner_model,
		refiner_finetuned=args.refiner_finetuned,
		max_examples=args.max_examples,
	)


if __name__ == "__main__":
	main()
