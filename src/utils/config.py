"""
YAML experiment config loader and validator.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate an experiment YAML config.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If required top-level keys are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    _validate(cfg, path)
    return cfg


_REQUIRED_KEYS = {"experiment_name", "features", "model"}


def _validate(cfg: Dict[str, Any], path: Path) -> None:
    missing = _REQUIRED_KEYS - set(cfg.keys())
    if missing:
        raise KeyError(
            f"Config {path.name} is missing required keys: {missing}"
        )

