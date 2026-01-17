"""Batch generation using vLLM for collecting model outputs."""

import json
import logging
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from llmhalluc.data import get_dataset_converter, load_data_config
from llmhalluc.hparams import GenerationArguments
from llmhalluc.utils import is_rank_zero, process_dataset

logger = logging.getLogger(__name__)


class GenerationExecutor:
    """Execute batch generation using vLLM."""

    def __init__(self, args: GenerationArguments):
        self.args = args
        self.llm = None
        self.tokenizer = None
        self.dataset = None
        self.results = []

    def setup(self):
        """Setup vLLM model and load dataset."""
        self._setup_model()
        self._setup_dataset()

    def _setup_model(self):
        """Initialize vLLM model."""
        from vllm import LLM

        logger.info(f"Loading model with vLLM: {self.args.model_name_or_path}")

        llm_kwargs = {
            "model": self.args.model_name_or_path,
            "tokenizer": self.args.tokenizer_name_or_path,
            "trust_remote_code": self.args.trust_remote_code,
            "tensor_parallel_size": self.args.tensor_parallel_size,
            "gpu_memory_utilization": self.args.gpu_memory_utilization,
            "seed": self.args.seed,
        }

        if self.args.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.args.max_model_len

        # Handle LoRA adapter
        if self.args.adapter_name_or_path:
            from vllm.lora.request import LoRARequest

            llm_kwargs["enable_lora"] = True
            self.lora_request = LoRARequest(
                "adapter", 1, self.args.adapter_name_or_path
            )
            logger.info(f"LoRA adapter enabled: {self.args.adapter_name_or_path}")
        else:
            self.lora_request = None

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("vLLM model loaded successfully")

    def _setup_dataset(self):
        data_config = load_data_config()

        dataset_info = data_config.get(self.args.dataset)
        if dataset_info is None:
            raise ValueError(
                f"Dataset '{self.args.dataset}' not found in dataset_info.json"
            )

        hf_url = dataset_info.get("hf_hub_url")
        if not hf_url:
            raise ValueError(
                f"Dataset '{self.args.dataset}' does not have 'hf_hub_url'"
            )

        # Use split from dataset_info.json, default to "train"
        split = dataset_info.get("split", "train")
        dataset = load_dataset(
            hf_url,
            name=dataset_info.get("subset"),
            split=split,
        )

        preprocess_converter_name = dataset_info.get("converter", None)
        column_mapping = dataset_info.get("columns", {})

        if preprocess_converter_name:
            preprocess_converter, preprocess_converter_args = get_dataset_converter(
                preprocess_converter_name, **column_mapping
            )
            dataset = process_dataset(
                dataset=dataset,
                processor=preprocess_converter,
                load_from_cache_file=getattr(self.args, "load_from_cache_file", True),
                **preprocess_converter_args,
            )

        # Apply SFT converter to match training format
        # SFTDatasetConverter produces {prompt: [messages], completion: [messages]}
        sft_converter, sft_converter_args = get_dataset_converter(
            "sft", **column_mapping if not preprocess_converter_name else {}
        )
        dataset = process_dataset(
            dataset=dataset,
            processor=sft_converter,
            load_from_cache_file=getattr(self.args, "load_from_cache_file", True),
            **sft_converter_args,
        )

        if self.args.max_samples is not None:
            dataset = dataset.select(range(min(self.args.max_samples, len(dataset))))

        self.dataset = dataset
        logger.info(
            f"Loaded dataset '{self.args.dataset}' with {len(self.dataset)} samples"
        )

    def _format_prompt(self, example: dict[str, Any]) -> str:
        # After SFTDatasetConverter, prompt is a list of messages
        # e.g., [{"role": "user", "content": "..."}]
        messages = example.get("prompt", [])

        if isinstance(messages, str):
            # Fallback: if prompt is still a string, wrap it
            messages = [{"role": "user", "content": messages}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Could not apply chat template: {e}")

        # Fallback: concatenate message contents
        return "\n".join(m.get("content", "") for m in messages)

    def _get_sampling_params(self):
        """Create vLLM SamplingParams from args."""
        from vllm import SamplingParams

        params = {
            "max_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature if self.args.do_sample else 0.0,
            "top_p": self.args.top_p,
            "n": self.args.num_return_sequences,
            "seed": self.args.seed,
        }

        if self.args.top_k > 0:
            params["top_k"] = self.args.top_k

        return SamplingParams(**params)

    def generate(self) -> list[dict[str, Any]]:
        """Run batch generation over the dataset."""

        sampling_params = self._get_sampling_params()
        prompts = [self._format_prompt(ex) for ex in self.dataset]

        logger.info(f"Starting generation for {len(prompts)} prompts")
        logger.info(
            f"Sampling params: temp={self.args.temperature}, "
            f"top_p={self.args.top_p}, max_tokens={self.args.max_new_tokens}"
        )

        # Print example
        if is_rank_zero() and len(prompts) > 0:
            self._print_example(prompts[0])

        # Generate in batches
        all_results = []
        batch_size = self.args.batch_size

        for batch_start in tqdm(
            range(0, len(prompts), batch_size),
            desc="Generating",
            disable=not is_rank_zero(),
        ):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            # Generate
            if self.lora_request:
                outputs = self.llm.generate(
                    batch_prompts,
                    sampling_params,
                    lora_request=self.lora_request,
                )
            else:
                outputs = self.llm.generate(batch_prompts, sampling_params)

            # Process outputs
            for i, output in enumerate(outputs):
                idx = batch_start + i
                example = self.dataset[idx]

                result = {
                    "idx": idx,
                    "prompt": example.get("prompt", ""),
                    "formatted_prompt": batch_prompts[i],
                    "generations": [o.text for o in output.outputs],
                }

                # Include reference (completion) if available
                if "completion" in example:
                    result["reference"] = example["completion"]

                # Include all original columns as metadata
                result["metadata"] = {
                    k: v
                    for k, v in example.items()
                    if k not in ["prompt", "completion"]
                }

                all_results.append(result)

            # Save intermediate results
            if batch_end % self.args.save_every < batch_size:
                self._save_results(all_results, intermediate=True)

        self.results = all_results
        return all_results

    def _print_example(self, prompt: str):
        """Print a formatted example similar to training."""
        print("\n" + "=" * 60)
        print("GENERATION EXAMPLE")
        print("=" * 60)
        print(f"\n[Formatted Prompt]:\n{prompt[:500]}...")
        print(f"\n[Model]: {self.args.model_name_or_path}")
        if self.args.adapter_name_or_path:
            print(f"[Adapter]: {self.args.adapter_name_or_path}")
        print(f"[Temperature]: {self.args.temperature}")
        print(f"[Max New Tokens]: {self.args.max_new_tokens}")
        print("=" * 60 + "\n")

    def _save_results(self, results: list[dict], intermediate: bool = False):
        """Save results to JSONL file."""
        # Ensure output directory exists
        output_dir = self.args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.args.results_path
        if intermediate:
            output_path = output_path.with_suffix(".intermediate.jsonl")

        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(results)} results to {output_path}")

    def save(self):
        """Save final results."""
        self._save_results(self.results, intermediate=False)

        # Also save a summary
        summary = {
            "model": self.args.model_name_or_path,
            "adapter": self.args.adapter_name_or_path,
            "dataset": self.args.dataset,
            "num_samples": len(self.results),
            "temperature": self.args.temperature,
            "max_new_tokens": self.args.max_new_tokens,
        }
        summary_path = self.args.results_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved summary to {summary_path}")


def run_gen(args: GenerationArguments) -> list[dict[str, Any]]:
    """Run batch generation.

    Args:
        args: Generation arguments.

    Returns:
        List of generation results.
    """
    executor = GenerationExecutor(args)
    executor.setup()
    results = executor.generate()
    executor.save()

    logger.info(f"Generation completed. {len(results)} samples processed.")
    return results
