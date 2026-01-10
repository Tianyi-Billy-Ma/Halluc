import argparse
from pathlib import Path


def str2bool(value):
    """Convert common string representations to bool for CLI parsing."""
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def path2str(value):
    if isinstance(value, Path):
        return str(value)
    return value


def normalized(value):
    value = path2str(value)
    return value
