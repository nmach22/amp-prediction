"""Abstract base class for all model wrappers."""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Minimal fit / predict interface shared by all models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Train the model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Targets of shape (n_samples,).

        Returns:
            self
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return model predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,).
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the positive class for classifiers.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Float array of shape (n_samples,) in [0, 1].

        Raises:
            NotImplementedError: If the model is not probabilistic or is a regressor.
        """
