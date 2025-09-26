"""Generic dataset processing script using converter architecture."""

import argparse
from pathlib import Path
from datasets import load_dataset

from llmhalluc.data import get_dataset_converter
from llmhalluc.utils import process_dataset


def main() -> None:
    """Main function for processing datasets using converters."""
    parser = argparse.ArgumentParser(description="Process datasets using converter architecture")
    parser.add_argument(
        "--converter", type=str, required=True, help="Name of the converter to use (e.g., 'squad', 'backtrack')"
    )
    parser.add_argument("--cache_dir", type=str, default="./.cache", help="Directory for caching datasets")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for saving processed datasets")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process")
    parser.add_argument(
        "--split", type=str, nargs="+", default=["train", "validation"], help="Dataset split(s) to process"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of times to repeat the dataset (only for train split)"
    )
    parser.add_argument("--num_proc", type=int, default=12, help="Number of processes for parallel processing")
    parser.add_argument("--redownload", action="store_true", help="Force redownload of the dataset")

    args = parser.parse_args()

    # Handle single split as string
    splits = args.split[0] if len(args.split) == 1 else args.split

    # Create converter (converters only process examples, not load/save datasets)
    converter = get_dataset_converter(args.converter)

    # Handle squad dataset loading from HuggingFace
    if args.converter == "squad" and args.dataset_name == "squad_v2":
        splits_list = splits if isinstance(splits, list) else [splits]

        for split in splits_list:
            # Load dataset from HuggingFace
            dataset = load_dataset(
                "rajpurkar/squad_v2",
                cache_dir=args.cache_dir,
                split=split,
                download_mode="force_redownload" if args.redownload else "reuse_dataset_if_exists",
            )

            # Process dataset using converter
            process_dataset(
                dataset=dataset,
                processor=converter,
                dataset_name="squad_v2",
                data_dir=args.data_dir,
                split=split,
                repeat=args.repeat,
                num_proc=args.num_proc,
            )
    else:
        # Generic processing for other converters (load from local JSON files)
        input_path = Path(args.data_dir) / args.dataset_name

        if isinstance(splits, list):
            for split in splits:
                dataset_path = input_path / f"{split}.json"
                if not dataset_path.exists():
                    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

                dataset = load_dataset("json", data_files=str(dataset_path))["train"]
                process_dataset(
                    dataset=dataset,
                    processor=converter,
                    dataset_name=f"{args.dataset_name}_{args.converter}",
                    data_dir=args.data_dir,
                    split=split,
                    repeat=args.repeat,
                    num_proc=args.num_proc,
                )
        else:
            dataset_path = input_path / f"{splits}.json"
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

            dataset = load_dataset("json", data_files=str(dataset_path))["train"]
            process_dataset(
                dataset=dataset,
                processor=converter,
                dataset_name=f"{args.dataset_name}_{args.converter}",
                data_dir=args.data_dir,
                split=splits,
                repeat=args.repeat,
                num_proc=args.num_proc,
            )


if __name__ == "__main__":
    main()
