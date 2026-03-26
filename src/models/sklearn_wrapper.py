"""
Thin wrapper that adapts any scikit-learn classifier to BaseModel.

Supported classifiers (configure via experiment YAML):
    random_forest     → RandomForestClassifier
    svm               → SVC  (with probability=True)
    logistic_regression → LogisticRegression
    gradient_boosting  → GradientBoostingClassifier
    knn               → KNeighborsClassifier

Example YAML block:
    model:
      name: random_forest
      params:
        n_estimators: 200
        max_depth: null
        random_state: 42
"""

from __future__ import annotations
from typing import Any, Dict
import numpy as np

from .base import BaseModel

_REGISTRY: Dict[str, Any] = {}


def _register():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    _REGISTRY.update(
        {
            "random_forest": RandomForestClassifier,
            "svm": lambda **kw: SVC(probability=True, **kw),
            "logistic_regression": LogisticRegression,
            "gradient_boosting": GradientBoostingClassifier,
            "knn": KNeighborsClassifier,
        }
    )


class SklearnModel(BaseModel):
    """Wrap a scikit-learn classifier as a BaseModel.

    Args:
        name: Registry key (see module docstring).
        params: Keyword arguments forwarded to the classifier constructor.
    """

    def __init__(self, name: str, params: Dict[str, Any] | None = None):
        if not _REGISTRY:
            _register()
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown model '{name}'. Available: {list(_REGISTRY)}"
            )
        self.name = name
        self.params = params or {}
        self._clf = _REGISTRY[name](**self.params)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnModel":
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)[:, 1]

