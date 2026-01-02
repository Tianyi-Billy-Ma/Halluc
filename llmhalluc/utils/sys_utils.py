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

