"""Utility helpers with lazy imports for optional dependencies."""

__all__ = [
    "load_config",
    "set_seed",
    "get_logger",
    "log_wandb_run",
    "resolve_wandb_settings",
]


def __getattr__(name):
    if name == "load_config":
        from .config import load_config

        return load_config
    if name == "set_seed":
        from .seed import set_seed

        return set_seed
    if name == "get_logger":
        from .logger import get_logger

        return get_logger
    if name == "log_wandb_run":
        from .wandb_logging import log_wandb_run

        return log_wandb_run
    if name == "resolve_wandb_settings":
        from .wandb_logging import resolve_wandb_settings

        return resolve_wandb_settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
