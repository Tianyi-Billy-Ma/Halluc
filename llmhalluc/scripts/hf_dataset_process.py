"""Dataset processing entrypoint leveraging the converter architecture.

Usage:
    python -m llmhalluc.scripts.hf_dataset_process \
        --converter backtrack \
        --hf_dataset_url openai/gsm8k \
        --subset main \
        --dataset_name gsm8k_backtrack \
        --cache_dir /tmp/hf-cache \
        --repeat 2 \
        --num_proc 16

Description:
    - Loads a Hugging Face dataset (optionally scoped to `--subset`) with an
      optional cache directory and redownload flag.
    - Instantiates the requested converter via `llmhalluc.data.get_dataset_converter`
      to transform individual examples into the project’s canonical format.
    - Passes the processed dataset to `llmhalluc.utils.process_dataset`, which
      supports dataset repetition for the train split, multiprocessing, and
      pushing artifacts to the hub via `--hf_push_url`.
"""

import argparse

from datasets import load_dataset

from llmhalluc.data import get_dataset_converter
from llmhalluc.utils import process_dataset


def main(arg_list: list[str] = None):
    """Main function for processing datasets using converters."""
    parser = argparse.ArgumentParser(
        description="Process datasets using converter architecture"
    )
    parser.add_argument(
        "--converter",
        type=str,
        required=True,
        help="Name of the converter to use (e.g., 'squad', 'backtrack')",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching datasets",
    )
    parser.add_argument(
        "--hf_dataset_url",
        type=str,
        required=True,
        help="URL of the dataset to process",
    )
    parser.add_argument(
        "--hf_push_url",
        type=str,
        default=None,
        help="Hugging Face Hub URL to push the dataset to",
    )
    parser.add_argument(
        "--subset", type=str, default="", help="Name of the dataset to upload"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat the dataset (only for train split)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=12,
        help="Number of processes for parallel processing",
    )
    parser.add_argument(
        "--redownload", action="store_true", help="Force redownload of the dataset"
    )

    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()

    # Create converter (converters only process examples, not load/save datasets)
    converter = get_dataset_converter(args.converter)

    dataset = load_dataset(
        args.hf_dataset_url,
        args.subset,
        cache_dir=args.cache_dir,
        download_mode="force_redownload"
        if args.redownload
        else "reuse_dataset_if_exists",
    )
    process_dataset(
        dataset=dataset,
        processor=converter,
        repeat=args.repeat,
        num_proc=args.num_proc,
        hf_push_url=args.hf_push_url,
    )


if __name__ == "__main__":
    main()
