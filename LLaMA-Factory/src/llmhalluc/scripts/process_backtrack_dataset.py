"""Backtrack dataset processing script - main novelty of the project."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_dataset

from llmhalluc.data import get_dataset_converter, BACKTRACK_TOKEN, BacktrackDatasetConverter
from llmhalluc.utils import process_dataset


def process_backtrack_dataset(
    dataset_name: str,
    converter: "BacktrackDatasetConverter",
    data_dir: str = "./data",
    split: str = "train",
    repeat: int = 1,
    num_proc: int = 12,
) -> None:
    """Process dataset with backtrack converter.

    Args:
        dataset_name: Name of the input dataset.
        converter: Configured backtrack converter instance.
        data_dir: Directory containing input and output datasets.
        split: Dataset split to process.
        repeat: Number of times to repeat dataset (only for train).
        num_proc: Number of processes for parallel processing.
    """
    # Load input dataset
    input_path = Path(data_dir) / dataset_name / f"{split}.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    dataset = load_dataset("json", data_files=str(input_path))["train"]

    # Process with backtrack converter
    processed_dataset = process_dataset(
        dataset=dataset,
        processor=converter,
        dataset_name=f"{dataset_name}_backtrack",
        split=split,
        repeat=repeat,
        num_proc=num_proc,
    )

    # # Save processed dataset
    save_path = Path(data_dir) / dataset_name / f"{split}.json"
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    processed_dataset.to_json(str(save_path), orient="records")


def main() -> None:
    """Main function for backtrack dataset processing."""
    parser = argparse.ArgumentParser(
        description="Process datasets with backtrack functionality - main project novelty"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./.cache", help="Directory for caching tokenizers and models"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Directory containing input and output datasets"
    )
    parser.add_argument("--dataset_name", type=str, default="squad_v2", help="Name of the input dataset")
    parser.add_argument(
        "--split", type=str, nargs="+", default=["train", "validation"], help="Dataset split(s) to process"
    )
    parser.add_argument(
        "--repeat", type=int, default=5, help="Number of times to repeat dataset (only for train split)"
    )
    parser.add_argument("--num_proc", type=int, default=12, help="Number of processes for parallel processing")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Tokenizer model name")
    parser.add_argument("--max_tokens", type=int, default=10, help="Maximum number of random tokens to add")

    args = parser.parse_args()

    # Load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir, trust_remote_code=True)
    tokenizer.add_tokens([BACKTRACK_TOKEN], special_tokens=True)

    # Prepare vocabulary for random token selection
    vocab = set(tokenizer.get_vocab().values())
    special_tokens = set(tokenizer.all_special_ids)
    no_spc_vocab = list(vocab - special_tokens)

    # Handle single split as string
    splits = args.split[0] if len(args.split) == 1 else args.split

    if isinstance(splits, list):
        for split in splits:
            # Create converter for each split
            converter = get_dataset_converter(
                "backtrack",
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
                no_spc_vocab=no_spc_vocab,
                split=split,
            )

            process_backtrack_dataset(
                dataset_name=args.dataset_name,
                converter=converter,
                data_dir=args.data_dir,
                split=split,
                repeat=args.repeat,
                num_proc=args.num_proc,
            )
    else:
        # Single split processing
        converter = get_dataset_converter(
            "backtrack",
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            no_spc_vocab=no_spc_vocab,
            split=splits,
        )

        process_backtrack_dataset(
            dataset_name=args.dataset_name,
            converter=converter,
            data_dir=args.data_dir,
            split=splits,
            repeat=args.repeat,
            num_proc=args.num_proc,
        )


if __name__ == "__main__":
    main()
