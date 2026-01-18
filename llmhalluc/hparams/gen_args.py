"""Generation arguments for batch inference with vLLM."""

from dataclasses import dataclass, field
from pathlib import Path

from llmhalluc.extras import OUTPUT_PATH

from .base_args import BaseArguments


@dataclass(kw_only=True)
class GenerationArguments(BaseArguments):
    run_name: str = "generation"
    stage: str = "sft"
    finetuning_type: str = "lora"
    exp_path: Path | None = None
    config_path: Path = field(init=False)

    model_name_or_path: str = ""
    tokenizer_name_or_path: str = ""
    adapter_name_or_path: str = ""
    trust_remote_code: bool = True

    dataset: str = ""
    max_samples: int | None = None

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = -1
    do_sample: bool = True
    num_return_sequences: int = 1

    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None

    output_path: str = ""
    output_filename: str = "generations.jsonl"
    save_every: int = 100

    seed: int = 42
    batch_size: int = 32

    # Derived field
    model_name: str = field(init=False)

    @property
    def yaml_exclude(self):
        return {
            "run_name",
            "exp_path",
            "config_path",
            "model_name",
        }

    def __post_init__(self):
        self.model_name = Path(self.model_name_or_path).name.lower()

        if self.exp_path is None:
            self.exp_path = (
                Path(OUTPUT_PATH)
                / self.model_name
                / self.run_name
                / self.stage
                / self.finetuning_type
            )

        self.output_path = str(self.exp_path / "gen")
        self.config_path = self.exp_path / "gen_config.yaml"

        if not self.tokenizer_name_or_path:
            self.tokenizer_name_or_path = self.model_name_or_path

        if self.tensor_parallel_size is None:
            import torch

            self.tensor_parallel_size = max(1, torch.cuda.device_count())

    @property
    def output_dir(self) -> Path:
        """Directory for generation outputs."""
        return Path(self.output_path)

    @property
    def results_path(self) -> Path:
        """Full path to results file."""
        return self.output_dir / self.output_filename
