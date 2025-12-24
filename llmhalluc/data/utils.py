from pathlib import Path
from llmhalluc.utils.cfg_utils import load_config

DEFAULT_DATASET_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "llmhalluc" / "dataset_config.yaml"
)


def load_data_config(
    path: str | Path = DEFAULT_DATASET_CONFIG_PATH,
) -> dict[str, any]:
    return load_config(path)
