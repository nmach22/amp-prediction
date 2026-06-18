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
    df = df[df["gram_status"].isin(GRAM_CLASSES)].reset_index(drop=True)
    return aggregate_duplicate_measurements(df)


def aggregate_duplicate_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated sequence-target MIC rows using median log10(MIC)."""
    group_columns = ["sequence", "target_activity_name"]
    if not set(group_columns).issubset(df.columns):
        return df.reset_index(drop=True)
    if not df.duplicated(group_columns).any():
        return df.reset_index(drop=True)

    first_rows = df.loc[~df.duplicated(group_columns, keep="first")]
    median_log_mic = (
        df.groupby(group_columns, sort=False)["log_mic"].median().to_numpy()
    )
    columns = {
        column: first_rows[column].to_numpy(copy=True)
        for column in df.columns
        if column not in {"activity", "log_mic"}
    }
    columns["activity"] = np.power(10.0, median_log_mic)
    columns["log_mic"] = median_log_mic
    return pd.DataFrame(columns, columns=df.columns)


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
        n_estimators: int = 3000,
        learning_rate: float = 0.02,
        max_depth: int = 4,
        min_child_weight: float = 5.0,
        subsample: float = 0.85,
        colsample_bytree: float = 0.75,
        reg_alpha: float = 0.1,
        reg_lambda: float = 5.0,
        early_stopping_rounds: int = 50,
    ):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for the xgboost_mic experiment. "
                "Install the project environment from env.yml."
            ) from exc

        self.random_state = random_state
        self._evals_result: dict = {}
        self._model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostMicRegressor":
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]
        if eval_set is None:
            early_stopping_rounds = self._model.get_params().get(
                "early_stopping_rounds"
            )
            if early_stopping_rounds is not None:
                self._model.set_params(early_stopping_rounds=None)
            try:
                self._model.fit(X, y, verbose=False)
            finally:
                if early_stopping_rounds is not None:
                    self._model.set_params(
                        early_stopping_rounds=early_stopping_rounds
                    )
            self._evals_result = {}
            return self

        self._model.fit(X, y, eval_set=eval_set, verbose=False)
        self._evals_result = self._model.evals_result()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "XGBoostMicRegressor is a regression model and does not expose "
            "class probabilities."
        )

    def artifact_metadata(self, feature_columns: list[str]) -> dict:
        """Return training diagnostics and feature importance for model artifacts."""
        booster = self._model.get_booster()
        importance = booster.get_score(importance_type="gain")
        return {
            "xgboost_best_iteration": _safe_model_attribute(
                self._model, "best_iteration"
            ),
            "xgboost_best_score": _safe_model_attribute(self._model, "best_score"),
            "xgboost_evals_result": self._evals_result,
            "xgboost_feature_importance_gain": {
                column: float(importance.get(column, 0.0))
                for column in feature_columns
            },
        }


def build_model(random_state: int = 42) -> XGBoostMicRegressor:
    """Create the XGBoost MIC regressor."""
    return XGBoostMicRegressor(random_state=random_state)


def _safe_model_attribute(model: object, name: str) -> object | None:
    try:
        return getattr(model, name)
    except AttributeError:
        return None


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
    "aggregate_duplicate_measurements",
    "build_model",
    "build_xgboost_features",
    "evaluate_taxonomy_predictions",
    "load_xgboost_mic_data",
    "xgboost_artifact_metadata",
]
