"""XGBoost MIC regression using sequence descriptors and taxonomy features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.sequence_descriptors import SequenceDescriptorEncoder
from src.models.base import BaseModel
from src.models.mic_baseline import GRAM_CLASSES
from src.models.taxonomy_mic_baseline import (
    TAXONOMY_FEATURE_PREFIXES,
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
    return build_xgboost_features_with_sequence_set(df, feature_set="full_modlamp")


def build_xgboost_features_with_sequence_set(
    df: pd.DataFrame,
    feature_set: str,
) -> pd.DataFrame:
    """Build XGBoost features using a named sequence descriptor set."""
    descriptor_encoder = SequenceDescriptorEncoder(feature_set=feature_set)
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


def build_xgboost_taxonomy_gram_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build target taxonomy and Gram-status features without sequence descriptors."""
    taxonomy_features = df[taxonomy_feature_columns(df)].astype(float)
    gram_features = _gram_features(df)
    return pd.concat(
        [
            taxonomy_features.reset_index(drop=True),
            gram_features.reset_index(drop=True),
        ],
        axis=1,
    )


def build_xgboost_sequence_only_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build sequence descriptors only, for ablation experiments."""
    descriptor_encoder = SequenceDescriptorEncoder(feature_set="full_modlamp")
    return descriptor_encoder.encode(df["sequence"])


def build_xgboost_basic_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build XGBoost features using only basic sequence descriptors."""
    return build_xgboost_features_with_sequence_set(df, feature_set="basic")


def build_xgboost_amp_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build XGBoost features using compact AMP-focused sequence descriptors."""
    return build_xgboost_features_with_sequence_set(df, feature_set="amp_core")


def build_xgboost_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build sequence, taxonomy, Gram, and peptide-target interaction features."""
    sequence_features = SequenceDescriptorEncoder(
        feature_set="interaction_core"
    ).encode(df["sequence"])
    taxonomy_features = df[taxonomy_feature_columns(df)].astype(float)
    gram_features = _gram_features(df)
    interaction_features = _sequence_target_interactions(
        sequence_features,
        gram_features,
        taxonomy_features,
    )
    return pd.concat(
        [
            sequence_features.reset_index(drop=True),
            taxonomy_features.reset_index(drop=True),
            gram_features.reset_index(drop=True),
            interaction_features.reset_index(drop=True),
        ],
        axis=1,
    )


def build_xgboost_motif_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build full physicochemical plus reduced-alphabet motif sequence features."""
    return build_xgboost_features_with_sequence_set(df, feature_set="motif_core")


def build_xgboost_selected_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build XGBoost features from full descriptors before train-only selection."""
    return build_xgboost_features_with_sequence_set(df, feature_set="full_modlamp")


def _gram_features(df: pd.DataFrame) -> pd.DataFrame:
    gram_features = pd.get_dummies(df["gram_status"], prefix="gram", dtype=float)
    for column in ["gram_gram_negative", "gram_gram_positive"]:
        if column not in gram_features:
            gram_features[column] = 0.0
    return gram_features[["gram_gram_negative", "gram_gram_positive"]]


def _sequence_target_interactions(
    sequence_features: pd.DataFrame,
    gram_features: pd.DataFrame,
    taxonomy_features: pd.DataFrame,
) -> pd.DataFrame:
    interaction_columns = [
        "modlamp_length",
        "modlamp_charge",
        "modlamp_charge_density",
        "modlamp_hydrophobic_ratio",
        "eisenberg_moment",
        "gravy_moment",
        "aa_frac_negative",
        "aa_frac_positive",
        "aa_frac_C",
        "local_max_positive_frac_w5",
        "local_max_hydrophobic_frac_w5",
        "longest_positive_run",
        "longest_hydrophobic_run",
    ]
    broad_taxonomy_columns = [
        column for column in taxonomy_features.columns if column.startswith("Phylum_")
    ]
    parts = []
    for sequence_column in interaction_columns:
        if sequence_column not in sequence_features:
            continue
        for gram_column in gram_features.columns:
            parts.append(
                pd.Series(
                    sequence_features[sequence_column].to_numpy()
                    * gram_features[gram_column].to_numpy(),
                    name=f"{sequence_column}_x_{gram_column}",
                )
            )
        for taxonomy_column in broad_taxonomy_columns:
            parts.append(
                pd.Series(
                    sequence_features[sequence_column].to_numpy()
                    * taxonomy_features[taxonomy_column].to_numpy(),
                    name=f"{sequence_column}_x_{taxonomy_column}",
                )
            )
    if not parts:
        return pd.DataFrame(index=sequence_features.index)
    return pd.concat(parts, axis=1)


def select_informative_feature_columns(
    X: pd.DataFrame,
    y: np.ndarray,
    top_k: int = 40,
    max_correlation: float = 0.95,
) -> list[str]:
    """Select sequence descriptors while preserving taxonomy and Gram features."""
    numeric = X.astype(float)
    protected_columns = [
        column
        for column in numeric.columns
        if column.startswith(TAXONOMY_FEATURE_PREFIXES)
        or column == "target_is_bacteria"
        or column.startswith("gram_")
    ]
    candidate_columns = [
        column for column in numeric.columns if column not in protected_columns
    ]
    variable_columns = [
        column
        for column in candidate_columns
        if numeric[column].nunique(dropna=False) > 1
    ]
    if not variable_columns:
        return protected_columns

    filtered = numeric[variable_columns]
    keep_columns: list[str] = []
    corr = filtered.corr().abs()
    for column in filtered.columns:
        if all(corr.loc[column, kept] <= max_correlation for kept in keep_columns):
            keep_columns.append(column)

    filtered = filtered[keep_columns]
    scores = np.abs(
        [
            np.corrcoef(filtered[column].to_numpy(), y)[0, 1]
            for column in filtered.columns
        ]
    )
    scores = np.nan_to_num(scores, nan=0.0)
    ranked = (
        pd.Series(scores, index=filtered.columns)
        .sort_values(ascending=False, kind="mergesort")
        .index.tolist()
    )
    selected_sequence_columns = ranked[: min(top_k, len(ranked))]
    return selected_sequence_columns + protected_columns


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
        "sequence_feature_set": "full_modlamp",
        "target": "log10_mic",
    }


def xgboost_basic_sequence_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return metadata for the basic sequence descriptor variant."""
    return {
        "taxonomy_feature_columns": taxonomy_feature_columns(df),
        "sequence_descriptor_columns": SequenceDescriptorEncoder(
            feature_set="basic"
        ).feature_names(),
        "descriptor_library": "modlamp",
        "sequence_feature_set": "basic",
        "target": "log10_mic",
    }


def xgboost_amp_core_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return metadata for the AMP-focused sequence descriptor variant."""
    return {
        "taxonomy_feature_columns": taxonomy_feature_columns(df),
        "sequence_descriptor_columns": SequenceDescriptorEncoder(
            feature_set="amp_core"
        ).feature_names(),
        "descriptor_library": "modlamp",
        "sequence_feature_set": "amp_core",
        "target": "log10_mic",
    }


def xgboost_interaction_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return metadata for the peptide-target interaction variant."""
    return {
        "taxonomy_feature_columns": taxonomy_feature_columns(df),
        "sequence_descriptor_columns": SequenceDescriptorEncoder(
            feature_set="interaction_core"
        ).feature_names(),
        "descriptor_library": "modlamp_plus_local_patterns",
        "sequence_feature_set": "interaction_core",
        "target": "log10_mic",
        "interaction_features": "sequence_by_gram_and_phylum",
    }


def xgboost_motif_sequence_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return metadata for the motif-aware sequence descriptor variant."""
    return {
        "taxonomy_feature_columns": taxonomy_feature_columns(df),
        "sequence_descriptor_columns": SequenceDescriptorEncoder(
            feature_set="motif_core"
        ).feature_names(),
        "descriptor_library": "modlamp_plus_reduced_alphabet_kmers",
        "sequence_feature_set": "motif_core",
        "target": "log10_mic",
    }


def xgboost_sequence_only_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return metadata for the sequence-only ablation."""
    return {
        "sequence_descriptor_columns": SequenceDescriptorEncoder().feature_names(),
        "descriptor_library": "modlamp_plus_local_patterns",
        "sequence_feature_set": "full_modlamp",
        "target": "log10_mic",
    }


def xgboost_taxonomy_gram_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return metadata for the taxonomy/Gram ablation."""
    return {
        "taxonomy_feature_columns": taxonomy_feature_columns(df),
        "target": "log10_mic",
        "target_features": "taxonomy_gram_only",
    }


__all__ = [
    "XGBoostMicRegressor",
    "aggregate_duplicate_measurements",
    "build_xgboost_amp_core_features",
    "build_xgboost_basic_sequence_features",
    "build_model",
    "build_xgboost_features",
    "build_xgboost_features_with_sequence_set",
    "build_xgboost_interaction_features",
    "build_xgboost_motif_sequence_features",
    "build_xgboost_selected_sequence_features",
    "build_xgboost_sequence_only_features",
    "build_xgboost_taxonomy_gram_features",
    "evaluate_taxonomy_predictions",
    "load_xgboost_mic_data",
    "select_informative_feature_columns",
    "xgboost_amp_core_artifact_metadata",
    "xgboost_artifact_metadata",
    "xgboost_basic_sequence_artifact_metadata",
    "xgboost_interaction_artifact_metadata",
    "xgboost_motif_sequence_artifact_metadata",
    "xgboost_sequence_only_artifact_metadata",
    "xgboost_taxonomy_gram_artifact_metadata",
]
