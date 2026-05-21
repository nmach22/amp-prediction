"""Utility helpers with lazy imports for optional dependencies."""

__all__ = ["load_config", "set_seed", "get_logger"]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
