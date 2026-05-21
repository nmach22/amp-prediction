"""Feature encoders.

Imports are lazy so one encoder's optional dependencies do not block unrelated
feature utilities.
"""

__all__ = [
    "OneHotEncoder",
    "PhysicochemicalEncoder",
    "Word2VecEncoder",
    "PLMEncoder",
]


def __getattr__(name):
    if name == "OneHotEncoder":
        from .onehot import OneHotEncoder

        return OneHotEncoder
    if name == "PhysicochemicalEncoder":
        from .physicochemical import PhysicochemicalEncoder

        return PhysicochemicalEncoder
    if name == "Word2VecEncoder":
        from .word2vec import Word2VecEncoder

        return Word2VecEncoder
    if name == "PLMEncoder":
        from .plm import PLMEncoder

        return PLMEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
