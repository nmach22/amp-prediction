from .base import BaseModel

__all__ = ["BaseModel", "SklearnModel"]


def __getattr__(name):
    if name == "SklearnModel":
        from .sklearn_wrapper import SklearnModel

        return SklearnModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
