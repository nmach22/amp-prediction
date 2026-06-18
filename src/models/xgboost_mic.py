"""XGBoost MIC regression using sequence descriptors and taxonomy features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.sequence_descriptors import SequenceDescriptorEncoder
from src.models.base import BaseModel
from src.models.mic_baseline import GRAM_CLASSES
from src.models.taxonomy_mic_baseline import (
    evaluate_taxonomy_predictions,
    load_taxonomy_mic_data,
    taxonomy_feature_columns,
)


def load_xgboost_mic_data(path: str) -> pd.DataFrame:
    """Load cleaned MIC rows for XGBoost training."""
    df = load_taxonomy_mic_data(path)
    if "gram_status" not in df.columns:
        raise ValueError("Missing required column: 'gram_status'")
    return df[df["gram_status"].isin(GRAM_CLASSES)].reset_index(drop=True)


def build_xgboost_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build sequence descriptor, taxonomy, and Gram-status features."""
    descriptor_encoder = SequenceDescriptorEncoder()
    sequence_features = descriptor_encoder.encode(df["sequence"])

    taxonomy_features = df[taxonomy_feature_columns(df)].astype(float)
    gram_features = pd.get_dummies(df["gram_status"], prefix="gram", dtype=float)
    for column in ["gram_gram_negative", "gram_gram_positive"]:
        if column not in gram_features:
            gram_features[column] = 0.0
    gram_features = gram_features[["gram_gram_negative", "gram_gram_positive"]]

    return pd.concat(
        [
            sequence_features.reset_index(drop=True),
            taxonomy_features.reset_index(drop=True),
            gram_features.reset_index(drop=True),
        ],
        axis=1,
    )


class XGBoostMicRegressor(BaseModel):
    """XGBoost regressor for log10(MIC)."""

    def __init__(
        self,
        random_state: int = 42,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for the xgboost_mic experiment. "
                "Install the project environment from env.yml."
            ) from exc

        self.random_state = random_state
        self._model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostMicRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "XGBoostMicRegressor is a regression model and does not expose "
            "class probabilities."
        )


def build_model(random_state: int = 42) -> XGBoostMicRegressor:
    """Create the XGBoost MIC regressor."""
    return XGBoostMicRegressor(random_state=random_state)


def xgboost_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return feature metadata stored with the trained artifact."""
    return {
        "taxonomy_feature_columns": taxonomy_feature_columns(df),
        "sequence_descriptor_columns": SequenceDescriptorEncoder().feature_names(),
        "descriptor_library": "modlamp",
        "target": "log10_mic",
    }


__all__ = [
    "XGBoostMicRegressor",
    "build_model",
    "build_xgboost_features",
    "evaluate_taxonomy_predictions",
    "load_xgboost_mic_data",
    "xgboost_artifact_metadata",
]
