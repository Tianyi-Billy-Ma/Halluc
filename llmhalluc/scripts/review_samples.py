import argparse
import glob
import json
import os
from typing import Any


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding line in {file_path}: {e}")
    return data


def find_sample_files(base_path: str) -> list[str]:
    """Find all samples_*.jsonl files in the given directory and subdirectories."""
    # If a specific file is given, return it
    if os.path.isfile(base_path) and base_path.endswith(".jsonl"):
        return [base_path]

    # Otherwise search recursively
    search_pattern = os.path.join(base_path, "**", "samples_*.jsonl")
    return glob.glob(search_pattern, recursive=True)


def print_sample(sample: dict[str, Any], index: int, total: int):
    """Print a single sample in a readable format."""
    print("=" * 80)
    print(f"Sample {index + 1}/{total}")
    print("=" * 80)

    # Print Metrics if available
    metrics = {
        k: v
        for k, v in sample.items()
        if k
        not in [
            "doc",
            "target",
            "arguments",
            "resps",
            "filtered_resps",
            "doc_hash",
            "prompt_hash",
            "target_hash",
            "doc_id",
            "filter",
            "metrics",
        ]
    }

    if metrics:
        print("\n[Metrics]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # Print Document/Input
    print("\n[Input/Document]")
    # Handle different doc structures (some might be strings, others dicts)
    if isinstance(sample.get("doc"), dict):
        print(json.dumps(sample["doc"], indent=2, ensure_ascii=False))
    else:
        print(sample.get("doc"))

    # Print Target/Reference
    print("\n[Target]")
    print(sample.get("target"))

    # Print Model Response(s)
    print("\n[Model Response]")
    resps = sample.get("resps", [])
    if isinstance(resps, list):
        for i, resp in enumerate(resps):
            # Arguments often contain the prompt, sometimes we might want to see it
            # But usually resps is what we care about
            print(f"--- Response {i + 1} ---")
            print(resp)
    else:
        print(resps)

    # Print Filtered Response (what the metric actually sees)
    filtered = sample.get("filtered_resps")
    if filtered:
        print("\n[Filtered Response]")
        if isinstance(filtered, list):
            for i, f_resp in enumerate(filtered):
                print(f"--- Filtered {i + 1} ---")
                print(f_resp)
        else:
            print(filtered)

    print("-" * 80)


def review_interactive(samples: list[dict[str, Any]]):
    """Interactive review mode."""
    if not samples:
        print("No samples to review.")
        return

    idx = 0
    total = len(samples)

    while True:
        print_sample(samples[idx], idx, total)

        print("\n[Commands] n: next, p: previous, q: quit, g <num>: go to index")
        cmd = input("Command: ").strip().lower()

        if cmd == "q":
            break
        elif cmd == "n":
            if idx < total - 1:
                idx += 1
            else:
                print("Already at the last sample.")
        elif cmd == "p":
            if idx > 0:
                idx -= 1
            else:
                print("Already at the first sample.")
        elif cmd.startswith("g "):
            try:
                target_idx = int(cmd.split()[1]) - 1
                if 0 <= target_idx < total:
                    idx = target_idx
                else:
                    print(f"Index must be between 1 and {total}")
            except ValueError:
                print("Invalid index.")
        elif cmd == "":
            # Default to next
            if idx < total - 1:
                idx += 1
            else:
                print("Already at the last sample.")
        else:
            print("Unknown command.")


def main():
    parser = argparse.ArgumentParser(description="Review lm_eval sample outputs.")
    parser.add_argument(
        "path",
        nargs="?",
        default="outputs",
        help="Path to samples file or directory containing samples.",
    )
    parser.add_argument(
        "--filter-incorrect",
        action="store_true",
        help="Only show samples where the model got it wrong (requires metrics).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="acc",
        help="Metric to use for filtering (default: acc).",
    )

    args = parser.parse_args()

    files = find_sample_files(args.path)

    if not files:
        print(f"No sample files found in {args.path}")
        # Try looking in default location if relative path failed
        if not os.path.isabs(args.path) and not args.path.startswith("outputs"):
            default_path = os.path.join("outputs", args.path)
            files = find_sample_files(default_path)
            if files:
                print(f"Found files in default output dir: {default_path}")

    if not files:
        print("Could not find any .jsonl files.")
        return

    # Select file if multiple
    selected_file = files[0]
    if len(files) > 1:
        print("Found multiple sample files:")
        for i, f in enumerate(files):
            print(f"{i + 1}. {f}")

        while True:
            try:
                choice = input("Select a file number (q to quit): ")
                if choice.lower() == "q":
                    return
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    selected_file = files[idx]
                    break
            except ValueError:
                pass

    print(f"Loading {selected_file}...")
    samples = load_jsonl(selected_file)
    print(f"Loaded {len(samples)} samples.")

    if args.filter_incorrect:
        # Try to find the metric key
        # Typical metrics: 'acc', 'exact_match', 'q_exact_match'
        # In the sample dict, metrics are keys with numeric values (usually 0.0 or 1.0 for binary)

        filtered_samples = []
        skipped = 0
        for s in samples:
            # Check if metric exists
            if args.metric in s:
                # Assuming 1.0 is correct and 0.0 is incorrect
                if s[args.metric] == 0:
                    filtered_samples.append(s)
            else:
                skipped += 1

        if skipped == len(samples):
            print(
                f"Warning: Metric '{args.metric}' not found in any samples. Showing all."
            )
        else:
            print(
                f"Filtered to {len(filtered_samples)} incorrect samples (metric: {args.metric})."
            )
            samples = filtered_samples

    review_interactive(samples)


if __name__ == "__main__":
    main()
