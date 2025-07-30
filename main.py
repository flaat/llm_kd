"""
LLM Experiment Runner

This script provides a command-line interface for running various language model experiments,
including testing models and generating counterfactual explanations. It provides options to:

1. Select model name, size and parameters
2. Configure generation parameters (temperature, top-p, etc.)
3. Choose the dataset for experiments
4. Run with plain or fine-tuned models

The script serves as the main entry point for the LLM experimentation framework.

Example usage:
    python main.py --model_name phi_4B --dataset adult --test_llm True
"""
import argparse
from typing import Dict, Any
from src.pipeline import test_llm, test_llm_refiner


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the experiment runner.
    
    Returns:
        Namespace containing all parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run language model experiments with configurable parameters."
    )

    # Model configuration parameters
    parser.add_argument(
        '--worker_model_name',
        type=str,
        default='phi_4B',
        help='Worker model name to use for experiments (default: phi_4B)'
    )

    parser.add_argument(
        '--refiner_model_name',
        type=str,
        default='phi_4B',
        help='Refiner model name to use for experiments (default: phi_4B)'
    )

    parser.add_argument(
        '--max_model_len',
        type=int,
        default=4000,
        help='Maximum context length for the model (default: 4000)'
    )
    
    parser.add_argument(
        '--fine_tuned',
        action='store_true',
        help='Use fine-tuned version of the model instead of base model'
    )

    parser.add_argument(
        '--refiner',
        action='store_true',
        help='Use refiner model for generating explanations'
    )

    # Generation parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Temperature for text generation - higher values increase randomness (default: 0.6)'
    )
    
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help='Top-p (nucleus) sampling parameter - lower values increase determinism (default: 0.8)'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=2048,
        help='Maximum number of tokens to generate (default: 2048)'
    )
    
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.05,
        help='Penalty applied to repeating tokens - higher values discourage repetition (default: 1.05)'
    )

    # Data and experiment type parameters
    parser.add_argument(
        '--dataset',
        type=str,
        default='cora',
        choices=['cora', 'adult', 'german', 'titanic'],
        help='Dataset to use for experiments (default: cora)'
    )
    
    parser.add_argument(
        '--test_llm',
        action='store_true',
        help='Run LLM testing pipeline'
    )

    return parser.parse_args()


def display_config(args: argparse.Namespace) -> None:
    """
    Display the experiment configuration parameters.
    
    Args:
        args: Parsed command-line arguments
    """
    print("\n========== EXPERIMENT CONFIGURATION ==========")
    print(f"Model name:          {args.model_name}")
    print(f"Model context length: {args.max_model_len}")
    print(f"Using fine-tuned:    {'Yes' if args.fine_tuned else 'No'}")
    print(f"Using refiner:      {'Yes' if args.refiner else 'No'}")
    print("\n----- Generation Parameters -----")
    print(f"Temperature:         {args.temperature}")
    print(f"Top-p:               {args.top_p}")
    print(f"Max tokens:          {args.max_tokens}")
    print(f"Repetition penalty:  {args.repetition_penalty}")
    print("\n----- Experiment Settings -----")
    print(f"Dataset:             {args.dataset}")
    print(f"Testing LLM:         {'Yes' if args.test_llm else 'No'}")
    print("=============================================\n")


def main() -> None:
    """
    Main entry point for the LLM experimentation framework.
    
    Parses command-line arguments, displays configuration, and runs
    the requested experiment pipeline based on argument values.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Display configuration for user verification
    display_config(args)
    
    # Run the selected experiment pipeline
    if args.test_llm:
        if args.refiner:
            test_llm_refiner(
                worker_model_name=args.worker_model_name,
                refiner_model_name=args.refiner_model_name,
                dataset=args.dataset,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                max_model_len=args.max_model_len,
                fine_tuned=args.fine_tuned
            )
        else:
            test_llm(
                model_name=args.worker_model_name,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                max_model_len=args.max_model_len,
                fine_tuned=args.fine_tuned
            )
    # Additional experiment types can be added here as elif branches


if __name__ == "__main__":
    main()
