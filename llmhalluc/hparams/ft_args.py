from dataclasses import dataclass, field

from trl import DPOConfig as BaseDPOConfig
from trl import GRPOConfig as BaseGRPOConfig
from trl import SFTConfig as BaseSFTConfig

from .base_args import BaseArguments


@dataclass
class FTArguments(BaseArguments):
    """Common arguments for fine-tuning tasks."""

    finetuning_type: str = field(
        default="lora", metadata={"help": "Type of fine-tuning to use"}
    )

    wandb_project: str = field(
        default="huggingface", metadata={"help": "WandB project name"}
    )

    config_path: str = field(default=None, metadata={"help": "Path to the config file"})

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier"}
    )
    tokenizer_name_or_path: str = field(
        default=None, metadata={"help": "Path to tokenizer or tokenizer identifier"}
    )
    dataset: str = field(
        default=None, metadata={"help": "The name of the dataset to use"}
    )
    eval_dataset: str | None = field(
        default=None, metadata={"help": "The name of the eval dataset to use"}
    )
    dataset_dir: str = field(
        default="./data", metadata={"help": "Path to the dataset directory"}
    )

    load_from_cache_file: bool = field(
        default=False, metadata={"help": "Whether to load from cache file"}
    )
    template: str = field(default="default", metadata={"help": "Chat template to use"})
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading model"},
    )

    # LoRA / QLoRA
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target: str = field(default="all", metadata={"help": "LoRA target modules"})
    load_in_4bit: bool = field(
        default=False, metadata={"help": "Whether to load model in 4-bit"}
    )

    # Special Tokens
    init_special_tokens: bool = field(
        default=False, metadata={"help": "Whether to initialize special tokens"}
    )
    new_special_tokens_config: dict[str, str] | None = field(
        default=None, metadata={"help": "New special tokens configuration"}
    )
    replace_text: dict[str, str] | None = field(
        default=None, metadata={"help": "Text replacement mapping"}
    )

    # Early Stopping
    early_stopping: bool = field(
        default=False, metadata={"help": "Whether to use early stopping"}
    )
    early_stopping_patience: int = field(
        default=3, metadata={"help": "Early stopping patience"}
    )
    early_stopping_threshold: float = field(
        default=0.001, metadata={"help": "Early stopping threshold"}
    )

    backtrack_token: str = field(
        default="<|BACKTRACK|>", metadata={"help": "Backtrack token"}
    )

    # Backtrack Training
    train_backtrack: bool = field(
        default=False,
        metadata={"help": "Enable backtrack training with error token masking"},
    )

    reset_position_ids: bool = field(
        default=False,
        metadata={"help": "Reset position IDs after backtrack tokens"},
    )

    dataset_download_mode: str | None = field(
        default=None,
        metadata={"help": "Dataset download mode (e.g., force_redownload)"},
    )


@dataclass
class SFTArguments(FTArguments, BaseSFTConfig):
    """Arguments for Supervised Fine-Tuning (SFT)."""

    run_name: str = field(default="sft")
    converter: str = field(
        default="sft",
        metadata={"help": "The converter to use for dataset conversion"},
    )

    def __post_init__(self):
        BaseSFTConfig.__post_init__(self)


@dataclass
class DPOArguments(FTArguments, BaseDPOConfig):
    """Arguments for Direct Preference Optimization (DPO)."""

    run_name: str = field(default="dpo")
    converter: str = field(
        default="dpo",
        metadata={"help": "The converter to use for dataset conversion"},
    )

    def __post_init__(self):
        BaseDPOConfig.__post_init__(self)


@dataclass
class GRPOArguments(FTArguments, BaseGRPOConfig):
    """Arguments for Group Relative Policy Optimization (GRPO)."""

    run_name: str = field(default="grpo")
    converter: str = field(
        default="grpo",
        metadata={"help": "The converter to use for dataset conversion"},
    )

    # GRPO Specific
    reward_funcs: str = field(
        default="", metadata={"help": "Comma-separated list of reward functions"}
    )
    reward_weights_str: str | None = field(
        default=None, metadata={"help": "Comma-separated list of reward weights"}
    )
    tokenize_labels: bool = field(
        default=True, metadata={"help": "Whether to tokenize labels"}
    )
    reward_func_args: dict[str, str | int | float | bool] = field(
        default_factory=lambda: {
            "outcome_weight": 1.0,
            "process_weight": 0.7,
            "backtrack_weight": 0.6,
            "format_weight": 0.3,
            "correction_bonus": 0.4,
            "unnecessary_penalty": 0.2,
            "efficiency_weight": 0.25,
            "failed_correction_penalty": 0.3,
            "use_curriculum": False,
            "enable_process_rewards": False,
            "enable_format_rewards": True,
            "max_backtracks": 20,
            "backtrack_token_id": None,
        },
        metadata={"help": "Arguments to pass to reward functions"},
    )

    # vLLM
    use_vllm: bool = field(
        default=True, metadata={"help": "Whether to use vLLM for generation"}
    )
    vllm_mode: str = field(
        default="colocate", metadata={"help": "vLLM mode: colocate or remote"}
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.3, metadata={"help": "vLLM GPU memory utilization"}
    )
    vllm_max_model_length: int | None = field(
        default=None, metadata={"help": "vLLM max model length"}
    )
    vllm_tensor_parallel_size: int = field(
        default=1, metadata={"help": "vLLM tensor parallel size"}
    )
    vllm_enable_sleep_mode: bool = field(
        default=False, metadata={"help": "vLLM enable sleep mode"}
    )
    vllm_server_host: str = field(
        default="0.0.0.0", metadata={"help": "vLLM server host"}
    )
    vllm_server_port: int = field(default=8000, metadata={"help": "vLLM server port"})
    vllm_server_timeout: float = field(
        default=240.0, metadata={"help": "vLLM server timeout"}
    )

    def __post_init__(self):
        BaseGRPOConfig.__post_init__(self)
        if not self.reward_funcs:
            raise ValueError(
                "reward_funcs is required for GRPO training. "
                "Provide comma-separated reward function names."
            )

    def get_reward_funcs_list(self) -> list[str]:
        """Parse reward_funcs string into list of names."""
        if not self.reward_funcs:
            return []
        return [n.strip() for n in self.reward_funcs.split(",") if n.strip()]

    def get_reward_weights_list(self) -> list[float] | None:
        """Parse reward_weights_str string into list of floats."""
        if not self.reward_weights_str:
            return None
        return [
            float(w.strip()) for w in self.reward_weights_str.split(",") if w.strip()
        ]
