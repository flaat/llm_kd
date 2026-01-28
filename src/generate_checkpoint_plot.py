#!/usr/bin/env python3
"""Generate professional checkpoint validation plot in the style of overall.pdf."""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.functions import (  # type: ignore  # noqa
    collect_validation_metrics,
    extract_checkpoint_name,
    compute_checkpoint_metrics,
)


def plot_checkpoint_metrics(
    rows: List[Dict],
    output_path: Path,
    dataset: str,
    model: str,
) -> None:
    """Generate professional line plot with dots for PFF and Parsing Rate across checkpoints.
    
    Highlights checkpoint 750 with a larger red dot to indicate the selected checkpoint.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        print(f"Warning: No data to plot for {dataset}/{model}")
        return

    # Sort rows by checkpoint number
    def get_checkpoint_num(row: Dict) -> int:
        checkpoint = row.get("checkpoint", "0")
        # Extract numeric part
        match = re.search(r"\d+", str(checkpoint))
        return int(match.group()) if match else 0

    sorted_rows = sorted(rows, key=get_checkpoint_num)
    checkpoints = [str(row["checkpoint"]) for row in sorted_rows]
    checkpoint_nums = [get_checkpoint_num(row) for row in sorted_rows]
    pff_values = [float(row.get("perfect_ff_rate", 0.0)) for row in sorted_rows]
    parsing_values = [float(row.get("parsing_rate", 0.0)) for row in sorted_rows]

    # Use the same professional style as overall.pdf
    plt.style.use("seaborn-v0_8-whitegrid")

    # Calculate figure width based on number of checkpoints to prevent overlap
    num_checkpoints = len(checkpoints)
    # Use wider figure for more checkpoints, minimum 14 inches
    fig_width = max(14, num_checkpoints * 0.4)
    
    # Create figure with 2 subplots side by side (PFF and Parsing Rate)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5))
    
    # Use a single color for the line (consistent with overall style)
    line_color = plt.cm.tab10(0)  # First color from tab10 palette
    x = np.arange(len(checkpoints))
    
    # Identify checkpoint 750 index (handle various formats like "750", "checkpoint_750", etc.)
    selected_checkpoint_num = 750
    selected_idx = None
    for idx, cp_num in enumerate(checkpoint_nums):
        if cp_num == selected_checkpoint_num:
            selected_idx = idx
            break
    
    # Plot PFF
    ax1 = axes[0]
    # Plot line with dots
    ax1.plot(x, pff_values, marker="o", color=line_color, alpha=0.9, 
             linewidth=2, markersize=6, label="PFF")
    
    # Highlight checkpoint 750 if it exists
    if selected_idx is not None:
        ax1.plot(x[selected_idx], pff_values[selected_idx], marker="o", 
                color="red", markersize=12, markeredgewidth=2, 
                markeredgecolor="darkred", zorder=10)
    
    ax1.set_title("PFF", fontsize=16, fontweight="bold")
    ax1.set_ylabel("PFF", fontsize=15)
    ax1.set_xlabel("Checkpoint", fontsize=15)
    ax1.set_xticks(x)
    # Rotate checkpoint labels and make them bigger, align with dots
    # Use 'center' alignment for better alignment with ticks
    ax1.set_xticklabels(checkpoints, rotation=45, ha="center", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.tick_params(axis="x", labelsize=12)
    # Set x-axis limits with padding to prevent label cutoff
    ax1.set_xlim(-0.5, len(checkpoints) - 0.5)
    
    # Plot Parsing Rate (JPR)
    ax2 = axes[1]
    # Plot line with dots
    ax2.plot(x, parsing_values, marker="o", color=line_color, alpha=0.9, 
             linewidth=2, markersize=6, label="JPR")
    
    # Highlight checkpoint 750 if it exists
    if selected_idx is not None:
        ax2.plot(x[selected_idx], parsing_values[selected_idx], marker="o", 
                color="red", markersize=12, markeredgewidth=2, 
                markeredgecolor="darkred", zorder=10)
    
    ax2.set_title("JPR", fontsize=16, fontweight="bold")
    ax2.set_ylabel("JPR", fontsize=15)
    ax2.set_xlabel("Checkpoint", fontsize=15)
    ax2.set_xticks(x)
    # Rotate checkpoint labels and make them bigger, align with dots
    # Use 'center' alignment for better alignment with ticks
    ax2.set_xticklabels(checkpoints, rotation=45, ha="center", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelsize=14)
    ax2.tick_params(axis="x", labelsize=12)
    # Set x-axis limits with padding to prevent label cutoff
    ax2.set_xlim(-0.5, len(checkpoints) - 0.5)

    # Adjust spacing between subplots to prevent overlap
    plt.tight_layout(pad=2.5, rect=[0, 0, 1, 0.98])

    # Save as PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    print(f"Checkpoint plot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate professional checkpoint validation plot"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., california)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., unsloth_qwen_0.5B)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/fine-tuning/worker_validation"),
        help="Base directory for validation results",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=200,
        help="Max examples per checkpoint file",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="checkpoint-image.pdf",
        help="Output filename (default: checkpoint-image.pdf)",
    )

    args = parser.parse_args()

    # Collect validation metrics for the specific dataset/model
    base_dir = args.base_dir
    rows = collect_validation_metrics(
        base_dir=base_dir,
        datasets=[args.dataset],
        models=[args.model],
        max_examples=args.max_examples,
    )

    if not rows:
        print(f"Error: No validation data found for {args.dataset}/{args.model}")
        print(f"Checked directory: {base_dir / args.dataset / args.model}")
        sys.exit(1)

    # Generate the plot
    output_path = base_dir / args.output_name
    plot_checkpoint_metrics(rows, output_path, args.dataset, args.model)


if __name__ == "__main__":
    main()
