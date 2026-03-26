"""
Abstract base class for all model wrappers.

Every model used in an experiment must implement this interface so that
run_experiment.py can call them uniformly, regardless of the underlying
framework (sklearn, torch, etc.).
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Minimal fit / predict interface shared by all models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Train the model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary labels of shape (n_samples,).

        Returns:
            self
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (0 / 1).

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Integer array of shape (n_samples,).
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the positive class.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Float array of shape (n_samples,) in [0, 1].
        """

