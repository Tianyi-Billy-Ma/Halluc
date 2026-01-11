"""Reward functions for GRPO training."""

from .base import BaseRewardFunction
from .bt import BacktrackRewardFunction
from .manager import (
    REWARD_FUNCTIONS,
    get_reward_function,
    get_reward_functions,
    list_reward_functions,
    register_reward,
)

# Register the backtrack reward function
register_reward("backtrack_grpo")(BacktrackRewardFunction)

__all__ = [
    "BaseRewardFunction",
    "BacktrackRewardFunction",
    "REWARD_FUNCTIONS",
    "get_reward_function",
    "get_reward_functions",
    "list_reward_functions",
    "register_reward",
]
