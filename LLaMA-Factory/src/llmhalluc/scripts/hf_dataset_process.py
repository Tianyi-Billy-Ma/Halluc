"""Generic dataset processing script using converter architecture."""

import argparse
from pathlib import Path
from datasets import load_dataset

from llmhalluc.data import get_dataset_converter
from llmhalluc.utils import process_dataset


def main(arg_list: list[str] = None):
    """Main function for processing datasets using converters."""
    parser = argparse.ArgumentParser(description="Process datasets using converter architecture")
    parser.add_argument(
        "--converter", type=str, required=True, help="Name of the converter to use (e.g., 'squad', 'backtrack')"
    )
    parser.add_argument("--cache_dir", type=str, default="./.cache", help="Directory for caching datasets")
    parser.add_argument("--hf_dataset_url", type=str, required=True, help="URL of the dataset to process")
    parser.add_argument("--subset", type=str, default="", help="Name of the dataset to upload")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to upload")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of times to repeat the dataset (only for train split)"
    )
    parser.add_argument("--num_proc", type=int, default=12, help="Number of processes for parallel processing")
    parser.add_argument("--redownload", action="store_true", help="Force redownload of the dataset")

    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()

    # Create converter (converters only process examples, not load/save datasets)
    converter = get_dataset_converter(args.converter)

    if args.subset:
        dataset = load_dataset(
            args.hf_dataset_url,
            args.subset,
            cache_dir=args.cache_dir,
            download_mode="force_redownload" if args.redownload else "reuse_dataset_if_exists",
        )
    else:
        dataset = load_dataset(
            args.hf_dataset_url,
            cache_dir=args.cache_dir,
            download_mode="force_redownload" if args.redownload else "reuse_dataset_if_exists",
        )

    processed_dataset = process_dataset(
        dataset=dataset,
        processor=converter,
        repeat=args.repeat,
        num_proc=args.num_proc,
    )
    processed_dataset.push_to_hub(args.dataset_name, private=True)


if __name__ == "__main__":
    main()
