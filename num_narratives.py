import argparse

from src.number_narratives import assess_narratives


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run number-of-narratives generation for coherence analysis"
    )

    parser.add_argument(
        '--worker_model_name',
        type=str,
        default='unsloth_qwen_0.5B',
        help='Model name to generate narratives (default: unsloth_qwen_0.5B)'
    )

    parser.add_argument(
        '--max_model_len',
        type=int,
        default=8192,
        help='Maximum context length for the model (default: 8192)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Temperature for text generation (default: 0.6)'
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help='Top-p sampling parameter (default: 0.8)'
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=4096,
        help='Maximum number of tokens to generate (default: 4096)'
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.05,
        help='Penalty applied to repeating tokens (default: 1.05)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='adult',
        choices=['adult', 'titanic', 'california', 'diabetes'],
        help='Dataset to use for experiments (default: adult)'
    )

    parser.add_argument(
        '--num_narratives',
        type=int,
        default=8,
        help='Number of narratives (K) to generate per sample (default: 8)'
    )

    return parser.parse_args()


def display_config(args: argparse.Namespace) -> None:
    print("\n========== NUMBER OF NARRATIVES GENERATION ==========")
    print(f"Model name:          {args.worker_model_name}")
    print(f"Model context length:{args.max_model_len}")
    print("\n----- Generation Parameters -----")
    print(f"Temperature:         {args.temperature}")
    print(f"Top-p:               {args.top_p}")
    print(f"Max tokens:          {args.max_tokens}")
    print(f"Repetition penalty:  {args.repetition_penalty}")
    print("\n----- Settings -----")
    print(f"Dataset:             {args.dataset}")
    print(f"Num narratives (K):  {args.num_narratives}")
    print("=============================================\n")


def main() -> None:
    args = parse_arguments()
    display_config(args)

    assess_narratives(
        model_name=args.worker_model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        dataset=args.dataset,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        max_model_len=args.max_model_len,
        num_narratives=args.num_narratives,
    )


if __name__ == "__main__":
    main()
