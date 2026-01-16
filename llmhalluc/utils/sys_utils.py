from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path_str
    return path.expanduser().resolve()


def is_dir(path: str | Path) -> bool:
    """Check if path is a directory (no file extension)."""
    return Path(path).suffix == ""


def is_rank_zero() -> bool:
    """Check if current process is rank 0 (main process).

    In distributed training, only rank 0 should perform logging and I/O
    to prevent duplicate outputs.

    Returns:
        True if rank 0 or not in distributed mode, False otherwise.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return True  # Single-process training
    return dist.get_rank() == 0
