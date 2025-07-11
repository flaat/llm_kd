import argparse
from src.build_dataset_refiner import build_dataset

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
        '--model_name',
        type=str,
        default='deepseek_r1_qwen_32B_Q4_AWQ1',
        help='Model name to use for experiments (default: deepseek_r1_qwen_32B_Q4_AWQ1)'
    )
    
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=4000,
        help='Maximum context length for the model (default: 4000)'
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
    print("\n----- Generation Parameters -----")
    print(f"Temperature:         {args.temperature}")
    print(f"Top-p:               {args.top_p}")
    print(f"Max tokens:          {args.max_tokens}")
    print(f"Repetition penalty:  {args.repetition_penalty}")
    print("\n----- Experiment Settings -----")
    print(f"Dataset:             {args.dataset}")
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

    # Run the dataset building and refinement process
    build_dataset(
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        dataset=args.dataset,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        max_model_len=args.max_model_len
    )
    

if __name__ == "__main__":
    main()