"""Generation arguments for batch inference with vLLM."""

from dataclasses import dataclass, field
from pathlib import Path

from .base_args import BaseArguments


@dataclass(kw_only=True)
class GenerationArguments(BaseArguments):
    """Arguments for batch generation using vLLM.

    Follows the same pattern as EvaluationArguments for integration
    with the e2e_cfg_setup pipeline.
    """

    # Experiment tracking (set by patch_configs)
    run_name: str
    exp_path: Path = field()
    config_path: Path = field(init=False)

    # Model
    model_name_or_path: str = ""
    tokenizer_name_or_path: str = ""
    adapter_name_or_path: str = ""
    trust_remote_code: bool = True

    # Dataset
    dataset: str = ""
    dataset_split: str = "train"
    max_samples: int | None = None
    prompt_column: str = "prompt"
    reference_column: str | None = "response"

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    do_sample: bool = True
    num_return_sequences: int = 1

    # vLLM settings
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None

    # Output
    output_path: str = ""
    output_filename: str = "generations.jsonl"
    save_every: int = 100

    # Misc
    seed: int = 42
    batch_size: int = 32

    @property
    def yaml_exclude(self):
        return {
            "run_name",
            "exp_path",
            "config_path",
        }

    def __post_init__(self):
        # Set output path based on exp_path
        self.output_path = str(self.exp_path / "gen")
        self.config_path = self.exp_path / "gen_config.yaml"

        # Default tokenizer to model path
        if not self.tokenizer_name_or_path:
            self.tokenizer_name_or_path = self.model_name_or_path

    @property
    def output_dir(self) -> Path:
        """Directory for generation outputs."""
        return Path(self.output_path)

    @property
    def results_path(self) -> Path:
        """Full path to results file."""
        return self.output_dir / self.output_filename
