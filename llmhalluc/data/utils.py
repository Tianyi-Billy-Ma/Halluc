import json
from pathlib import Path

DEFAULT_DATASET_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "dataset_info.json"
)


def load_data_config(
    path: str | Path = DEFAULT_DATASET_CONFIG_PATH,
) -> dict[str, any]:
    """Load dataset configuration from JSON file.

    Args:
        path: Path to dataset config file (JSON format)

    Returns:
        Dictionary containing dataset configurations
    """
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)
