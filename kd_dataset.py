import argparse
from src.build_dataset import build_dataset_ref, build_dataset_wor

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
        default='unsloth_qwen3_1.7B',
        help='Worker model name to generate draft narratives (default: unsloth_qwen3_1.7B for with-refiner pipeline, else qwen3_30B_A3B)'
    )
    
    parser.add_argument(
        '--refiner_model_name',
        type=str,
        default='qwen3_30B_A3B',
        help='Refiner model name to refine draft narratives (default: qwen3_30B_A3B)'
    )
    
    parser.add_argument(
        '--max_model_len',  
        type=int,
        default=8192,
        help='Maximum context length for the model (default: 8192)'
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
        default=5000,
        help='Maximum number of tokens to generate (default: 4096)'
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
        default='adult',
        choices=['adult','titanic', 'california', 'diabetes'],
        help='Dataset to use for experiments (default: adult)'
    )

    parser.add_argument(
        '--number_narratives',
        type=int,
        default=5,
        help='Number of narratives to generate (default: 10)'
    )

    parser.add_argument(
        '--only_worker',
        action='store_true',
        help='Whether to only run the worker model (default: False)'
    )

    parser.add_argument(
        '--fine_tuned',
        action='store_true',
        help='Use LoRA fine-tuned worker model (default: False)'
    )

    parser.add_argument(
        '--lora_checkpoint_path',
        type=str,
        default=None,   
        help='Path to LoRA checkpoint when using --fine_tuned (default: None)'
    )

    return parser.parse_args()


def display_config(args: argparse.Namespace) -> None:
    """
    Display the experiment configuration parameters.
    
    Args:
        args: Parsed command-line arguments
    """
    print("\n========== EXPERIMENT CONFIGURATION ==========")
    print(f"Worker Model name:    {args.worker_model_name}")
    print(f"Refiner Model name:   {args.refiner_model_name}")
    print(f"Model context length: {args.max_model_len}")
    print("\n----- Generation Parameters -----")
    print(f"Temperature:         {args.temperature}")
    print(f"Top-p:               {args.top_p}")
    print(f"Max tokens:          {args.max_tokens}")
    print(f"Repetition penalty:  {args.repetition_penalty}")
    print("\n----- Experiment Settings -----")
    print(f"Dataset:             {args.dataset}")
    print(f"Number of narratives: {args.number_narratives}")
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

    if args.only_worker:
        # Run only the worker model dataset building process
        build_dataset_wor(
            model_name=args.worker_model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            dataset=args.dataset,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            max_model_len=args.max_model_len,
            fine_tuned=args.fine_tuned,
            lora_checkpoint_path=args.lora_checkpoint_path,
        )
    else:
        # Run the dataset building and refinement process
        build_dataset_ref(
            worker_model_name=args.worker_model_name,
            refiner_model_name=args.refiner_model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            dataset=args.dataset,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            max_model_len=args.max_model_len,
            number_narratives=args.number_narratives,
            fine_tuned=args.fine_tuned,
            lora_checkpoint_path=args.lora_checkpoint_path,
        )
    

if __name__ == "__main__":
    main()