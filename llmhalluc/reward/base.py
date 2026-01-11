"""Base reward function for GRPO training."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from transformers import TrainerState


@dataclass
class BaseRewardFunction(ABC):
    """Base class for reward functions.

    Reward functions must implement __call__ to compute rewards for completions.
    The reward function receives prompts, completions, and any additional dataset
    columns as keyword arguments.

    Following TRL's reward function interface:
    https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function

    Requirements:
    1. Accept prompts, completions, completions_ids, trainer_state as kwargs
    2. Accept any additional dataset columns as kwargs (use **kwargs)
    3. Return a list of floats (one reward per completion)
    4. Can return None if reward is not applicable (for multi-task training)

    Attributes:
        name: Identifier for the reward function (used in logging)
        weight: Weight for combining with other reward functions (default: 1.0)
    """

    name: str = "base_reward"
    weight: float = 1.0

    @property
    def __name__(self) -> str:
        """Return the name of the reward function (for TRL compatibility)."""
        return self.name

    @abstractmethod
    def __call__(
        self,
        prompts: list[str] | list[list[dict[str, str]]],
        completions: list[str] | list[list[dict[str, str]]],
        completions_ids: list[list[int]] | None = None,
        trainer_state: TrainerState | None = None,
        **kwargs: Any,
    ) -> list[float] | None:
        """Compute rewards for the given completions.

        Args:
            prompts: List of prompts (either plain text or conversational format)
            completions: List of completions generated for each prompt
            completions_ids: Tokenized completion IDs (for token-level rewards)
            trainer_state: Current trainer state (for curriculum learning)
            **kwargs: Additional dataset columns (e.g., ground_truth, labels)

        Returns:
            List of float rewards, one per completion.
            Return None if reward is not applicable to these samples.
        """
        pass

    @property
    def processing_class(self) -> Any | None:
        """Return the processing class for this reward function.

        Only needed when reward function is a model-based reward.
        For custom reward functions, this should return None.

        Returns:
            PreTrainedTokenizer or None
        """
        return None
