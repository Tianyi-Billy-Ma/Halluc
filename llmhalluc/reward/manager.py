"""Reward function registry and manager for GRPO training."""

import inspect

from .base import BaseRewardFunction

# Registry of available reward functions
# Keys are string identifiers used in config files
# Values are reward function classes
REWARD_FUNCTIONS: dict[str, type[BaseRewardFunction]] = {}


def register_reward(name: str):
    """Decorator to register a reward function.

    Usage:
        @register_reward("accuracy")
        class AccuracyReward(BaseRewardFunction):
            ...

    Args:
        name: Unique identifier for the reward function

    Returns:
        Decorator function
    """

    def decorator(cls: type[BaseRewardFunction]) -> type[BaseRewardFunction]:
        if name in REWARD_FUNCTIONS:
            raise ValueError(f"Reward function '{name}' is already registered.")
        REWARD_FUNCTIONS[name] = cls
        return cls

    return decorator


def get_reward_function(name: str, **kwargs: any) -> BaseRewardFunction:
    """Get a reward function instance by name.

    Args:
        name: Name of the registered reward function
        **kwargs: Arguments to pass to the reward function constructor

    Returns:
        Instantiated reward function

    Raises:
        ValueError: If reward function name not found
    """
    if name not in REWARD_FUNCTIONS:
        available = ", ".join(REWARD_FUNCTIONS.keys()) or "none"
        raise ValueError(
            f"Reward function '{name}' not found. "
            f"Available reward functions: {available}"
        )

    reward_cls = REWARD_FUNCTIONS[name]

    # Inspect constructor signature to pass only relevant arguments
    sig = inspect.signature(reward_cls.__init__)
    params = sig.parameters

    # Check if constructor accepts **kwargs
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    if accepts_kwargs:
        return reward_cls(**kwargs)

    # Filter kwargs to match constructor arguments
    valid_kwargs = {k: v for k, v in kwargs.items() if k in params}
    return reward_cls(**valid_kwargs)


def get_reward_functions(
    names: str | list[str],
    **kwargs: any,
) -> list[BaseRewardFunction]:
    """Get multiple reward function instances from comma-separated names.

    Args:
        names: Comma-separated string of reward function names, or list of names
        **kwargs: Arguments to pass to all reward function constructors

    Returns:
        List of instantiated reward functions

    Example:
        >>> get_reward_functions("accuracy,format")
        [AccuracyReward(...), FormatReward(...)]
    """
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]

    return [get_reward_function(name, **kwargs) for name in names]


def list_reward_functions() -> list[str]:
    """List all registered reward function names.

    Returns:
        List of reward function identifiers
    """
    return list(REWARD_FUNCTIONS.keys())
