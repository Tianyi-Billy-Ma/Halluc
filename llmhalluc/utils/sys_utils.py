from omegaconf import OmegaConf
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path_str
    return path.expanduser().resolve()


def load_config(path: str) -> dict[str, any]:
    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    return OmegaConf.to_container(cfg, resolve=True) or {}


def apply_overrides(config: dict[str, any], overrides: list[str]) -> dict[str, any]:
    if not overrides:
        return dict(config)

    base_conf = OmegaConf.create(config)
    override_conf = OmegaConf.from_dotlist(overrides)
    merged = OmegaConf.merge(base_conf, override_conf)
    return OmegaConf.to_container(merged, resolve=True)


def save_config(args: dict[str, any], path: str | Path) -> None:
    cfg_path = resolve_path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(args), cfg_path)
