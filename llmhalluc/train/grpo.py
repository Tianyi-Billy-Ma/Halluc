"""GRPO training executor for TRL GRPOTrainer."""

import logging

from llmhalluc.hparams.ft_args import GRPOArguments
from llmhalluc.reward import get_reward_functions

from .base import BaseExecutor
from .callbacks import get_callbacks

logger = logging.getLogger(__name__)


class GRPOExecutor(BaseExecutor):
    """Executor for Group Relative Policy Optimization (GRPO) training.

    GRPO is an online RL algorithm that generates multiple completions
    per prompt and uses group-relative rewards to compute advantages.

    This executor handles:
    - Model and tokenizer setup (inherited from BaseExecutor)
    - Dataset loading and conversion to GRPO format
    - Reward function instantiation from registry
    - GRPOTrainer setup with PEFT support
    - vLLM integration for efficient generation (default: colocate mode)
    """

    def __init__(self, args: GRPOArguments):
        super().__init__(args)

    def _get_dataset_converter(self):
        """Return GRPODatasetConverter for group relative policy optimization."""
        return "grpo"

    def _get_reward_functions(self):
        """Initialize reward functions from the registry.

        Parses the reward_funcs argument (comma-separated names) and
        instantiates each reward function from REWARD_FUNCTIONS registry.
        """
        reward_func_names = self.args.get_reward_funcs_list()
        if not reward_func_names:
            raise ValueError(
                "No reward functions specified. "
                "Set reward_funcs in config (comma-separated names)."
            )

        logger.info(f"Loading reward functions: {reward_func_names}")

        reward_args = (self.args.reward_func_args or {}).copy()

        # Auto-inject backtrack_token_id if not present
        if (
            "backtrack_token_id" not in reward_args
            or reward_args["backtrack_token_id"] is None
        ):
            # Try to resolve backtrack_token token
            backtrack_token = self.args.backtrack_token
            token_id = self.tokenizer.convert_tokens_to_ids(backtrack_token)

            # Verify if token exists (convert_tokens_to_ids returns int, check against unk if strictly needed,
            # but usually unique special tokens are added explicitly)
            if token_id != self.tokenizer.unk_token_id:
                reward_args["backtrack_token_id"] = token_id
                logger.info(f"Auto-detected backtrack_token_id={token_id}")
            else:
                logger.warning(
                    f"Token '{backtrack_token}' not found in tokenizer. "
                    "Reward functions relying on it may fail if backtrack_token_id is not set."
                )

        reward_funcs = get_reward_functions(reward_func_names, **reward_args)

        # Log reward weights if specified
        reward_weights = self.args.get_reward_weights_list()
        if reward_weights:
            if len(reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward_weights ({len(reward_weights)}) must match "
                    f"number of reward_funcs ({len(reward_funcs)})"
                )
            logger.info(f"Reward weights: {reward_weights}")

        # Log vLLM configuration
        if self.args.use_vllm:
            logger.info(
                f"vLLM enabled: mode={self.args.vllm_mode}, "
                f"gpu_memory_utilization={self.args.vllm_gpu_memory_utilization}"
            )
        else:
            logger.info("vLLM disabled, using model.generate() for completions")
        return reward_funcs

    def _get_trainer_class(self):
        """Return GRPOTrainer class for GRPO training."""
        from trl import GRPOTrainer

        return GRPOTrainer

    def _get_dataset(self, dataset_key: str, split_type: str = "train"):
        """Load and process dataset with tokenizer for GRPO.

        Overrides base class to pass tokenizer to GRPODatasetConverter
        when tokenize_ground_truth=True.
        """
        from datasets import load_dataset

        from llmhalluc.data import get_dataset_converter, load_data_config
        from llmhalluc.utils import process_dataset, wrap_converter_with_replace

        data_config = load_data_config()

        dataset_info = data_config.get(dataset_key)
        if dataset_info is None:
            raise ValueError(f"Dataset '{dataset_key}' not found in dataset_info.json")

        hf_url = dataset_info.get("hf_hub_url")
        if not hf_url:
            raise ValueError(f"Dataset '{dataset_key}' does not have 'hf_hub_url'")

        default_split = "train" if split_type == "train" else "test"
        split = dataset_info.get("split", default_split)

        dataset = load_dataset(
            hf_url,
            name=dataset_info.get("subset"),
            split=split,
            download_mode=getattr(self.args, "dataset_download_mode", None),
        )

        # Build converter kwargs with tokenizer if tokenize_labels is enabled
        converter_kwargs = dataset_info.get("columns", {}).copy()
        if getattr(self.args, "tokenize_labels", True):
            converter_kwargs["tokenizer"] = self.tokenizer

        converter, converter_args = get_dataset_converter(
            self.args.converter, **converter_kwargs
        )

        if getattr(self.args, "replace_text", None):
            converter = wrap_converter_with_replace(
                converter,
                self.args.replace_text,
                converter_args.get("batched", False),
            )

        dataset = process_dataset(
            dataset=dataset,
            processor=converter,
            split=split,
            load_from_cache_file=self.args.load_from_cache_file,
            **converter_args,
        )

        return dataset

    def setup_trainer(self):
        """Setup GRPOTrainer with model, tokenizer, dataset, and reward functions.

        Unlike SFT/DPO trainers, GRPO requires:
        - reward_funcs: List of reward functions (callables)
        """
        trainer_class = self._get_trainer_class()

        # Build LoRA config if using PEFT
        peft_config = self._get_peft_config()

        # Get callable reward functions
        reward_funcs = self._get_reward_functions()

        # Build callbacks
        callbacks = get_callbacks(self.args)

        trainer_kwargs = {
            "model": self.model,
            "args": self.args,
            "processing_class": self.tokenizer,
            "train_dataset": self.dataset.get("train"),
            "eval_dataset": self.dataset.get("eval"),
            "reward_funcs": reward_funcs,
            "peft_config": peft_config,
            "callbacks": callbacks if callbacks else None,
        }

        self.trainer = trainer_class(**trainer_kwargs)


def run_grpo(args: GRPOArguments):
    """Run GRPO training with the given arguments.

    Args:
        args: GRPOArguments containing all training configuration.
    """
    executor = GRPOExecutor(args=args)
    executor.on_train_start()
    executor.fit()
    executor.on_train_end()
