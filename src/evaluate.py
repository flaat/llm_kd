from pathlib import Path
from typing import Optional, Dict, Any

from .functions import (
    build_input_path,
    build_output_dir,
    compute_metrics_for_dataset,
    export_results,
    generate_result_row,
    generate_latex_table,
    load_json,
)


def evaluate_single(
    dataset_name: str,
    worker_model: str,
    *,
    refiner: bool = False,
    worker_finetuned: bool = False,
    refiner_model: Optional[str] = None,
    refiner_finetuned: bool = False,
    max_examples: int = 200,
) -> Dict[str, Any]:
    """
    Evaluate one dataset/model (with or without refiner) and return metrics + row.
    """
    input_path = build_input_path(
        dataset_name=dataset_name,
        worker_model=worker_model,
        refiner=refiner,
        worker_finetuned=worker_finetuned,
        refiner_model=refiner_model,
        refiner_finetuned=refiner_finetuned,
    )
    data = load_json(input_path)
    metrics = compute_metrics_for_dataset(data, max_examples=max_examples, dataset_name=dataset_name)
    row = generate_result_row(
        dataset_name=dataset_name,
        worker_model=worker_model,
        refiner=refiner,
        metrics=metrics,
        worker_finetuned=worker_finetuned,
        refiner_model=refiner_model,
        refiner_finetuned=refiner_finetuned,
    )
    latex = generate_latex_table([row], refiner=refiner)
    return {"row": row, "latex": latex, "metrics": metrics, "input_path": str(input_path)}


def evaluate_and_save(
    dataset_name: str,
    worker_model: str,
    *,
    refiner: bool = False,
    worker_finetuned: bool = False,
    refiner_model: Optional[str] = None,
    refiner_finetuned: bool = False,
    max_examples: int = 200,
) -> Dict[str, Any]:
    """
    Evaluate and persist results (JSON + LaTeX) into {dataset_name}/{model_name} directory.
    """
    result = evaluate_single(
        dataset_name=dataset_name,
        worker_model=worker_model,
        refiner=refiner,
        worker_finetuned=worker_finetuned,
        refiner_model=refiner_model,
        refiner_finetuned=refiner_finetuned,
        max_examples=max_examples,
    )

    base_output_dir = build_output_dir(refiner=refiner)
    worker_suffix = "ft" if worker_finetuned else "plain"
    
    # Build model name: for refiner it's worker--refiner, otherwise just worker
    if refiner:
        model_name = f"{worker_model}--{refiner_model}"
        ref_suffix = "ft" if refiner_finetuned else "plain"
        file_stub = f"results_{worker_model}-{worker_suffix}_{refiner_model}-{ref_suffix}"
    else:
        model_name = worker_model
        file_stub = f"results_{worker_model}-{worker_suffix}"
    
    # Output directory: {base}/{dataset_name}/{model_name}
    output_dir = base_output_dir / dataset_name / model_name

    export_results(
        [result["row"]],
        refiner=refiner,
        output_dir=output_dir,
        json_name=f"{file_stub}.json",
        tex_name=f"{file_stub}.tex",
    )
    return result
