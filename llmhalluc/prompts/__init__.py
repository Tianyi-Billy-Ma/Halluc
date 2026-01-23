"""Prompt registry for dataset converters.

This module provides a registry of prompts that can be referenced by name
in dataset configurations. When a dataset doesn't have a 'prompt' column,
converters can fall back to using a registered prompt.

Usage:
    from llmhalluc.prompts import get_prompt, PROMPT_REGISTRY

    # Get a registered prompt by name
    prompt = get_prompt("math")  # Returns MATH_INSTRUCTION

    # Check if a prompt is registered
    if "math" in PROMPT_REGISTRY:
        ...
"""

from .MathPrompt import MATH_INSTRUCTION, MATH_PROMPT_TEMPLATE
from .QAPrompt import QA_INSTRUCTION, QA_PROMPT_TEMPLATE

# Registry mapping prompt names to prompt strings
PROMPT_REGISTRY: dict[str, str] = {
    "math": MATH_INSTRUCTION.strip(),
    "math_template": MATH_PROMPT_TEMPLATE.strip(),
    "qa": QA_INSTRUCTION.strip(),
    "qa_template": QA_PROMPT_TEMPLATE.strip(),
}


def get_prompt(name: str) -> str:
    """Get a registered prompt by name.

    Args:
        name: Name of the prompt in the registry.

    Returns:
        The prompt string.

    Raises:
        ValueError: If the prompt name is not found in the registry.
    """
    if name not in PROMPT_REGISTRY:
        available = ", ".join(sorted(PROMPT_REGISTRY.keys()))
        raise ValueError(
            f"Prompt '{name}' not found in registry. Available prompts: {available}"
        )
    return PROMPT_REGISTRY[name]


def register_prompt(name: str, prompt: str) -> None:
    """Register a new prompt.

    Args:
        name: Name to register the prompt under.
        prompt: The prompt string.

    Raises:
        ValueError: If a prompt with this name already exists.
    """
    if name in PROMPT_REGISTRY:
        raise ValueError(
            f"Prompt '{name}' already exists. Use a different name or update the existing prompt."
        )
    PROMPT_REGISTRY[name] = prompt.strip()


def list_prompts() -> list[str]:
    """List all registered prompt names.

    Returns:
        Sorted list of registered prompt names.
    """
    return sorted(PROMPT_REGISTRY.keys())
