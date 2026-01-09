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
	collect_global_feasibility_results,
	generate_global_feasibility_latex_table,
	generate_global_feasibility_barplots,
	generate_global_fra_latex_table,
	generate_global_fra_barplots,
	collect_number_narratives_metrics,
	plot_number_narratives_metrics,
	collect_overall_metrics,
	plot_overall_metrics,
)


# ------------------------- CLI -------------------------

def parse_args():
	parser = argparse.ArgumentParser(description="Generate results (global) or validation metrics.")
	parser.add_argument("--validation", action="store_true", help="Run validation mode on results/fine-tuning/worker_validation.")
	parser.add_argument("--global", action="store_true", dest="global_mode", help="Generate global results by scanning all model directories.")
	parser.add_argument("--overall", action="store_true", help="Average global metrics across multiple datasets.")
	parser.add_argument("--number-narratives", action="store_true", dest="number_narratives", help="Run number-of-narratives NCS analysis.")
	parser.add_argument("--datasets", nargs="+", default=["adult", "titanic", "california", "diabetes"], help="Datasets to include for validation.")
	parser.add_argument("--models", nargs="+", default=[
		"unsloth_qwen_0.5B",
		#"unsloth_qwen3_0.6B",
		#"unsloth_llama_1B-Instruct",
		#"unsloth_deepseek_r1_qwen_1.5B",
		#"unsloth_qwen3_1.7B",
		#"unsloth_llama_3B-Instruct",
		#"unsloth_qwen_3B",
		#"unsloth_qwen3_4B",
	], help="Models to include for validation.")
	parser.add_argument("--max-examples", type=int, default=200, help="Max examples per checkpoint file.")
	parser.add_argument("--output-dir", type=Path, default=None, help="Root directory for validation outputs; defaults to base_dir/<dataset>/<model>.")

	# Number narratives specific args
	parser.add_argument("--num-narratives", type=int, default=8, help="K value (max narratives generated per sample).")
	parser.add_argument("--n-start", type=int, default=3, help="Minimum N for subset sampling.")
	parser.add_argument("--alpha", type=float, default=0.6, help="Alpha parameter for coherence score.")

	# Single-run (non-validation) convenience flags
	parser.add_argument("--dataset-name", type=str, help="Dataset name for single run (non-validation).")
	parser.add_argument("--worker-model", type=str, help="Worker model name for single run.")
	parser.add_argument("--refiner", action="store_true", help="Enable refiner mode (for validation or single run). For validation, worker and refiner models are the same, so provide single model names.")
	parser.add_argument("--worker-finetuned", action="store_true", help="Worker finetuned flag for single run.")
	parser.add_argument("--refiner-model", type=str, help="Refiner model name for single run (required if --refiner).")
	parser.add_argument("--refiner-finetuned", action="store_true", help="Refiner finetuned flag for single run.")
	return parser.parse_args()


def main():
	args = parse_args()

	if args.validation:
		if not args.models:
			raise SystemExit("Provide --models for validation mode.")
		
		# Use refiner_validation directory if --refiner flag is set
		if args.refiner:
			base_dir = Path("results/fine-tuning/refiner_validation")
			print(f"Using refiner validation mode. Base directory: {base_dir}")
			print(f"Note: Worker and refiner models are the same. Converting model names to 'model_name--model_name' format.")
			# Convert model names to worker--refiner format (where worker == refiner)
			refiner_models = [f"{model}--{model}" for model in args.models]
		else:
			base_dir = Path("results/fine-tuning/worker_validation")
			print(f"Using worker validation mode. Base directory: {base_dir}")
			refiner_models = args.models
		
		rows = collect_validation_metrics(base_dir, datasets=args.datasets, models=refiner_models, max_examples=args.max_examples)
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

	# Number narratives mode
	if args.number_narratives:
		if not args.datasets:
			raise SystemExit("Provide --datasets for number-narratives mode.")
		if not args.models:
			raise SystemExit("Provide --models for number-narratives mode.")
		
		base_dir = Path("results/number_narratives")
		
		# Use multiple alpha values
		alphas = [0.5, 0.6, 0.8]
		
		print(f"Collecting NCS metrics for datasets: {args.datasets}, models: {args.models}")
		print(f"Parameters: K={args.num_narratives}, n_start={args.n_start}, alphas={alphas}")
		
		metrics = collect_number_narratives_metrics(
			base_dir=base_dir,
			datasets=args.datasets,
			models=args.models,
			num_narratives=args.num_narratives,
			n_start=args.n_start,
			alphas=alphas
		)
		
		# Generate plots for each dataset in subfolders
		base_output_dir = base_dir / "plots"
		base_output_dir.mkdir(parents=True, exist_ok=True)
		
		for dataset in args.datasets:
			if dataset in metrics and metrics[dataset]:
				# Create dataset-specific subfolder
				dataset_output_dir = base_output_dir / dataset
				dataset_output_dir.mkdir(parents=True, exist_ok=True)
				
				# Plot NCS for each alpha
				for alpha in alphas:
					metric_key = f"ncs_alpha_{alpha}"
					plot_path = dataset_output_dir / f"ncs_alpha_{alpha}.png"
					plot_number_narratives_metrics(
						metrics[dataset],
						plot_path,
						dataset,
						metric_key=metric_key,
						metric_label=f"NCS (α={alpha})",
						n_start=args.n_start,
						num_narratives=args.num_narratives
					)
				
				# Plot Jaccard similarity
				plot_path = dataset_output_dir / "jaccard.png"
				plot_number_narratives_metrics(
					metrics[dataset],
					plot_path,
					dataset,
					metric_key="jaccard",
					metric_label="Jaccard Similarity",
					n_start=args.n_start,
					num_narratives=args.num_narratives
				)
				
				# Plot Ranking similarity
				plot_path = dataset_output_dir / "ranking.png"
				plot_number_narratives_metrics(
					metrics[dataset],
					plot_path,
					dataset,
					metric_key="ranking",
					metric_label="Ranking Similarity (Kendall τ)",
					n_start=args.n_start,
					num_narratives=args.num_narratives
				)
				
				# Also save metrics to JSON
				metrics_json_path = dataset_output_dir / "metrics.json"
				# Convert int keys to str for JSON serialization
				serializable_metrics = {}
				for model, model_data in metrics[dataset].items():
					serializable_metrics[model] = {str(k): v for k, v in model_data.items()}
				with metrics_json_path.open("w") as f:
					json.dump(serializable_metrics, f, indent=2)
				print(f"Metrics saved to: {metrics_json_path}")
			else:
				print(f"Warning: No metrics found for dataset '{dataset}'")
		
		return

	# Global mode (scan all models)
	if args.global_mode:
		if args.overall:
			if not args.datasets:
				raise SystemExit("Provide --datasets for overall mode.")
			overall = collect_overall_metrics(args.datasets, args.refiner)
			if not overall:
				raise SystemExit("No metrics found for the requested datasets.")

			output_dir = build_output_dir(args.refiner) / "global_results"
			output_path = output_dir / "overall.png"
			plot_overall_metrics(overall, args.datasets, args.refiner, output_path)
			print(f"Overall plot saved to: {output_path}")
			return

		if not args.dataset_name:
			raise SystemExit("Provide --dataset-name for global mode.")
		
		try:
			results = collect_global_results(args.dataset_name, args.refiner)
		except ValueError as e:
			raise SystemExit(str(e))
		if not results:
			raise SystemExit(f"No results found for dataset '{args.dataset_name}' in {'with_refiner' if args.refiner else 'draft_generator'} directory.")
		
		# Generate LaTeX table (global metrics)
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
		
		# Generate and save barplots (global metrics)
		png_filename = f"{args.dataset_name}_{suffix}.png"
		png_path = output_dir / png_filename
		generate_global_barplots(results, args.dataset_name, args.refiner, png_path)
		print(f"Barplots saved to: {png_path}")

		# Generate feasibility LaTeX table and barplots (mean/std)
		try:
			feasibility_results = collect_global_feasibility_results(args.dataset_name, args.refiner)
		except ValueError as e:
			print(f"Warning: {e}")
			feasibility_results = {}
		
		if feasibility_results:
			feas_tex_filename = f"{args.dataset_name}_{suffix}_feasibility.tex"
			feas_tex_path = output_dir / feas_tex_filename
			feasibility_latex = generate_global_feasibility_latex_table(
				feasibility_results, args.dataset_name, args.refiner
			)
			with feas_tex_path.open("w") as f:
				f.write(feasibility_latex)
			print(f"Feasibility LaTeX table saved to: {feas_tex_path}")
			
			feas_png_filename = f"{args.dataset_name}_{suffix}_feasibility.png"
			feas_png_path = output_dir / feas_png_filename
			generate_global_feasibility_barplots(
				feasibility_results, args.dataset_name, args.refiner, feas_png_path
			)
			print(f"Feasibility barplots saved to: {feas_png_path}")
		else:
			print(f"Warning: No feasibility results found for dataset '{args.dataset_name}'.")
		
		# Generate FRA-only LaTeX table
		fra_latex_table = generate_global_fra_latex_table(results, args.dataset_name, args.refiner)
		fra_tex_filename = f"{args.dataset_name}_{suffix}_fra.tex"
		fra_tex_path = output_dir / fra_tex_filename
		with fra_tex_path.open("w") as f:
			f.write(fra_latex_table)
		print(f"FRA LaTeX table saved to: {fra_tex_path}")

		# Generate and save FRA-only barplots (2x2 grid, two plots per row)
		fra_png_filename = f"{args.dataset_name}_{suffix}_fra.png"
		fra_png_path = output_dir / fra_png_filename
		generate_global_fra_barplots(results, args.dataset_name, args.refiner, fra_png_path)
		print(f"FRA barplots saved to: {fra_png_path}")
		
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
