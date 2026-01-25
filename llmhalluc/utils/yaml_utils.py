"""YAML utilities replacing OmegaConf functionality.

Provides YAML loading, saving, and config merging without antlr4 dependency.
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML file and return as dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save dictionary to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict.

    - Nested dicts are merged recursively
    - Override values replace base values for non-dict types
    - Base is not modified; returns a new dict
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def parse_value(value: str) -> Any:
    """Parse string value to appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in ("none", "null", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def from_dotlist(dotlist: list[str]) -> dict[str, Any]:
    """Convert OmegaConf-style dotlist to nested dictionary.

    Example:
        ['model.lr=1e-4', 'train.epochs=10'] -> {'model': {'lr': 1e-4}, 'train': {'epochs': 10}}
    """
    result: dict[str, Any] = {}
    for item in dotlist:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = parse_value(value)
    return result


def to_container(data: Any) -> Any:
    """Convert nested structures to plain Python containers.

    Handles dict, list, and passes through other types.
    """
    if isinstance(data, dict):
        return {k: to_container(v) for k, v in data.items()}
    if isinstance(data, list):
        return [to_container(v) for v in data]
    return data
