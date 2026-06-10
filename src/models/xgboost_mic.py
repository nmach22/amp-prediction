"""XGBoost MIC regression using sequence descriptors and taxonomy features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.plm import (
    DEFAULT_ESM2_MODEL,
    DEFAULT_MIC_EMBEDDING_PATH,
    embeddings_for_sequences,
    load_embedding_cache,
    load_embedding_cache_metadata,
)
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


def build_xgboost_esm2_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build frozen ESM2 peptide embeddings plus taxonomy and Gram features."""
    embeddings = embeddings_for_sequences(
        df["sequence"].astype(str).tolist(),
        DEFAULT_MIC_EMBEDDING_PATH,
    )
    embedding_features = pd.DataFrame(
        embeddings,
        columns=[f"esm2_{index}" for index in range(embeddings.shape[1])],
    )
    taxonomy_features = df[taxonomy_feature_columns(df)].astype(float)
    gram_features = _gram_features(df)
    return pd.concat(
        [
            embedding_features.reset_index(drop=True),
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


def pca_reduce_esm2_features(
    X_train: pd.DataFrame,
    validation_features: dict[str, pd.DataFrame],
    y_train: np.ndarray,
    n_components: int = 128,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Fit PCA on train ESM2 columns and apply it to validation features."""
    del y_train
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    esm2_columns = [column for column in X_train.columns if column.startswith("esm2_")]
    if not esm2_columns:
        raise ValueError("No ESM2 feature columns found for PCA reduction.")
    context_columns = [column for column in X_train.columns if column not in esm2_columns]
    components = min(n_components, len(esm2_columns), len(X_train))
    if components < 1:
        raise ValueError("PCA requires at least one training row and one ESM2 column.")

    scaler = StandardScaler()
    pca = PCA(n_components=components, random_state=42)
    train_esm2 = scaler.fit_transform(X_train[esm2_columns].astype(float))
    train_pca = pca.fit_transform(train_esm2)

    X_train_reduced = _join_pca_context(
        train_pca,
        X_train[context_columns],
        components,
    )
    transformed_validation = {}
    for split_name, X_split in validation_features.items():
        split_esm2 = scaler.transform(X_split[esm2_columns].astype(float))
        split_pca = pca.transform(split_esm2)
        transformed_validation[split_name] = _join_pca_context(
            split_pca,
            X_split[context_columns],
            components,
        )

    metadata = {
        "feature_transform": "train_only_standard_scaler_pca_on_esm2",
        "esm2_original_dim": len(esm2_columns),
        "esm2_pca_components": components,
        "esm2_pca_explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
        "esm2_pca_explained_variance_total": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "passthrough_feature_columns": context_columns,
        "esm2_feature_columns": list(esm2_columns),
        "esm2_pca_scaler": scaler,
        "esm2_pca_model": pca,
    }
    return X_train_reduced, transformed_validation, metadata


def _join_pca_context(
    pca_values: np.ndarray,
    context: pd.DataFrame,
    n_components: int,
) -> pd.DataFrame:
    pca_features = pd.DataFrame(
        pca_values,
        columns=[f"esm2_pca_{index}" for index in range(n_components)],
        index=context.index,
    )
    return pd.concat(
        [
            pca_features.reset_index(drop=True),
            context.reset_index(drop=True).astype(float),
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


def build_regularized_esm2_model(random_state: int = 42) -> XGBoostMicRegressor:
    """Create a stronger-regularized XGBoost regressor for dense ESM2 features."""
    return XGBoostMicRegressor(
        random_state=random_state,
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=2,
        min_child_weight=20.0,
        subsample=0.65,
        colsample_bytree=0.35,
        reg_alpha=1.0,
        reg_lambda=25.0,
        early_stopping_rounds=100,
    )


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


def xgboost_esm2_context_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return metadata for frozen ESM2 plus target-context features."""
    metadata = load_embedding_cache_metadata(DEFAULT_MIC_EMBEDDING_PATH)
    _, embeddings = load_embedding_cache(DEFAULT_MIC_EMBEDDING_PATH)
    return {
        "taxonomy_feature_columns": taxonomy_feature_columns(df),
        "embedding_model": metadata.get("model_name", DEFAULT_ESM2_MODEL),
        "embedding_path": str(DEFAULT_MIC_EMBEDDING_PATH),
        "embedding_dim": int(embeddings.shape[1]),
        "target": "log10_mic",
        "target_features": "frozen_esm2_taxonomy_gram",
    }


__all__ = [
    "XGBoostMicRegressor",
    "aggregate_duplicate_measurements",
    "build_xgboost_amp_core_features",
    "build_xgboost_basic_sequence_features",
    "build_model",
    "build_regularized_esm2_model",
    "build_xgboost_esm2_context_features",
    "build_xgboost_features",
    "build_xgboost_features_with_sequence_set",
    "build_xgboost_interaction_features",
    "build_xgboost_motif_sequence_features",
    "build_xgboost_selected_sequence_features",
    "build_xgboost_sequence_only_features",
    "build_xgboost_taxonomy_gram_features",
    "evaluate_taxonomy_predictions",
    "load_xgboost_mic_data",
    "pca_reduce_esm2_features",
    "select_informative_feature_columns",
    "xgboost_amp_core_artifact_metadata",
    "xgboost_artifact_metadata",
    "xgboost_basic_sequence_artifact_metadata",
    "xgboost_esm2_context_artifact_metadata",
    "xgboost_interaction_artifact_metadata",
    "xgboost_motif_sequence_artifact_metadata",
    "xgboost_sequence_only_artifact_metadata",
    "xgboost_taxonomy_gram_artifact_metadata",
]
