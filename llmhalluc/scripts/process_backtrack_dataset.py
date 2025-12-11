"""Backtrack dataset processing script - main novelty of the project."""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llmhalluc.data.backtrack import (
    BACKTRACK_TOKEN,
)
from llmhalluc.data import get_dataset_converter
from llmhalluc.utils.data_utils import process_dataset


def main(arg_list: list[str] = None) -> None:
    """Main function for backtrack dataset processing."""
    parser = argparse.ArgumentParser(
        description="Process datasets with backtrack functionality - main project novelty"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching tokenizers and models",
    )
    parser.add_argument(
        "--converter",
        type=str,
        default="backtrack",
        help="Name of the converter to use (e.g., 'GSM8KSymbolicDatasetConverter')",
    )
    parser.add_argument(
        "--hf_dataset_url",
        type=str,
        required=True,
        help="URL of the dataset to process",
    )
    parser.add_argument(
        "--subset", type=str, default="", help="Name of the dataset to upload"
    )
    parser.add_argument(
        "--hf_push_url", type=str, default=None, help="Name of the input dataset"
    )
    parser.add_argument(
        "--split", type=str, nargs="+", default=None, help="Dataset split(s) to process"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Number of times to repeat dataset (only for train split)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=12,
        help="Number of processes for parallel processing",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Tokenizer model name",
    )
    parser.add_argument(
        "--redownload", action="store_true", help="Force redownload of the dataset"
    )
    parser.add_argument(
        "--option",
        choices=["all", "half", "random", "single"],
        default="half",
        help="Option for backtracking (all, half, random, single)",
    )
    parser.add_argument(
        "--prop",
        type=float,
        default=0.5,
        help="Proportion of examples to to extend. Only valid when option 'single' is selected. ",
    )

    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()

    # Load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer.add_tokens([BACKTRACK_TOKEN], special_tokens=True)

    # Handle single split as string
    splits = args.split[0] if args.split and len(args.split) == 1 else args.split

    converter = get_dataset_converter(
        args.converter,
        tokenizer=tokenizer,
        option=args.option,
        prop=args.prop,
    )

    dataset = load_dataset(
        args.hf_dataset_url,
        # args.subset,
        cache_dir=args.cache_dir,
        download_mode="force_redownload"
        if args.redownload
        else "reuse_dataset_if_exists",
    )

    dataset["train"] = dataset["test"]

    process_dataset(
        dataset=dataset,
        processor=converter,
        split=splits,
        repeat=args.repeat,
        num_proc=args.num_proc,
        hf_push_url=args.hf_push_url,
    )


if __name__ == "__main__":
    arg_list = [
        "--converter",
        "gsm8k_symbolic_backtrack",
        "--hf_dataset_url",
        "apple/GSM-Symbolic",
        "--hf_push_url",
        "GSM8K-Symbolic-Backtrack-all",
        "--subset",
        "main",
        "--num_proc",
        "12",
        "--repeat",
        "5",
    ]
    main(arg_list)
