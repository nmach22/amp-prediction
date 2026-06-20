"""CatBoost MIC regression with physicochemical sequence engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.sequence_descriptors import SequenceDescriptorEncoder
from src.models.base import BaseModel
from src.models.mic_baseline import GRAM_CLASSES, NONSTANDARD_PATTERN
from src.models.taxonomy_mic_baseline import evaluate_taxonomy_predictions

TAXONOMY_RANK_COLUMNS = ("Phylum", "Class", "Order", "Family", "Genus")
CATBOOST_CATEGORICAL_COLUMNS = (
    "target_activity_name",
    "gram_status",
    *TAXONOMY_RANK_COLUMNS,
)
DESCRIPTOR_FEATURE_SET = "motif_core"
ENGINEERED_FEATURE_COLUMNS = (
    "eng_charge_per_length",
    "eng_abs_charge_per_length",
    "eng_hydrophobic_to_abs_charge",
    "eng_positive_minus_negative_frac",
    "eng_aromatic_to_hydrophobic",
    "eng_boman_per_length",
    "eng_moment_balance",
    "eng_local_charge_span",
    "eng_length_bin",
)


def load_catboost_mic_data(path: str) -> pd.DataFrame:
    """Load and clean MIC rows for CatBoost training."""
    df = pd.read_csv(path)
    required = {"sequence", "target_activity_name", "activity", "gram_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["sequence", "target_activity_name", "activity"])
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.upper().str.strip()
    cleaned["target_activity_name"] = (
        cleaned["target_activity_name"].astype(str).str.strip()
    )
    cleaned["gram_status"] = cleaned["gram_status"].astype(str).str.strip()
    cleaned["activity"] = pd.to_numeric(cleaned["activity"], errors="coerce")
    cleaned = cleaned.dropna(subset=["activity"])
    cleaned = cleaned[cleaned["activity"] > 0]
    cleaned = cleaned[cleaned["sequence"].str.len() > 0]
    cleaned = cleaned[~cleaned["sequence"].str.contains(NONSTANDARD_PATTERN)]
    cleaned = cleaned[cleaned["gram_status"].isin(GRAM_CLASSES)].copy()

    for column in TAXONOMY_RANK_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = "Unknown"
        cleaned[column] = (
            cleaned[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    cleaned["log_mic"] = np.log10(cleaned["activity"])
    return aggregate_duplicate_measurements(cleaned.reset_index(drop=True))


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
    return pd.DataFrame(columns, columns=df.columns).reset_index(drop=True)


def build_catboost_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build descriptor, engineered, Gram, and target taxonomy features."""
    descriptor_encoder = SequenceDescriptorEncoder(feature_set=DESCRIPTOR_FEATURE_SET)
    sequence_features = descriptor_encoder.encode(df["sequence"])
    engineered = build_engineered_physchem_features(sequence_features)
    categorical = _categorical_features(df)
    return pd.concat(
        [
            sequence_features.reset_index(drop=True),
            engineered.reset_index(drop=True),
            categorical.reset_index(drop=True),
        ],
        axis=1,
    )


def build_engineered_physchem_features(
    sequence_features: pd.DataFrame,
) -> pd.DataFrame:
    """Create target-free descriptor transforms for MIC regression."""
    length = _safe_denominator(sequence_features["modlamp_length"])
    abs_charge = _safe_denominator(sequence_features["modlamp_charge"].abs())
    hydrophobic = sequence_features["aa_frac_hydrophobic"]
    positive = sequence_features["aa_frac_positive"]
    negative = sequence_features["aa_frac_negative"]

    engineered = pd.DataFrame(
        {
            "eng_charge_per_length": sequence_features["modlamp_charge"] / length,
            "eng_abs_charge_per_length": sequence_features["modlamp_charge"].abs()
            / length,
            "eng_hydrophobic_to_abs_charge": hydrophobic / abs_charge,
            "eng_positive_minus_negative_frac": positive - negative,
            "eng_aromatic_to_hydrophobic": sequence_features["aa_frac_aromatic"]
            / _safe_denominator(hydrophobic),
            "eng_boman_per_length": sequence_features["modlamp_boman"] / length,
            "eng_moment_balance": sequence_features["eisenberg_moment"]
            - sequence_features["gravy_moment"],
            "eng_local_charge_span": sequence_features["local_max_positive_frac_w5"]
            - sequence_features["local_max_negative_frac_w5"],
        }
    )
    length_bins = pd.cut(
        sequence_features["modlamp_length"],
        bins=[0, 10, 20, 40, np.inf],
        labels=False,
        include_lowest=True,
    )
    engineered["eng_length_bin"] = length_bins.astype(float)
    return engineered.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical = pd.DataFrame(index=df.index)
    for column in CATBOOST_CATEGORICAL_COLUMNS:
        if column in df.columns:
            values = df[column]
        else:
            values = pd.Series("Unknown", index=df.index)
        categorical[column] = (
            values.fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
    return categorical


def _safe_denominator(values: pd.Series) -> pd.Series:
    return values.astype(float).abs().clip(lower=1e-8)


class CatBoostMicRegressor(BaseModel):
    """CatBoost regressor for log10(MIC)."""

    def __init__(
        self,
        random_state: int = 42,
        iterations: int = 4000,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 10.0,
        random_strength: float = 1.0,
        subsample: float = 0.85,
        rsm: float = 1.0,
        bootstrap_type: str = "MVS",
        loss_function: str = "RMSE",
        eval_metric: str = "MAE",
        early_stopping_rounds: int = 100,
    ):
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "catboost is required for the catboost_mic_physchem experiment. "
                "Install the project environment from env.yml."
            ) from exc

        self.random_state = random_state
        self.numeric_medians_: dict[str, float] = {}
        self.categorical_columns_ = list(CATBOOST_CATEGORICAL_COLUMNS)
        self.numeric_columns_: list[str] = []
        self.feature_columns_: list[str] = []
        self._model = CatBoostRegressor(
            loss_function=loss_function,
            eval_metric=eval_metric,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            subsample=subsample,
            rsm=rsm,
            bootstrap_type=bootstrap_type,
            early_stopping_rounds=early_stopping_rounds,
            random_seed=random_state,
            allow_writing_files=False,
            verbose=False,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "CatBoostMicRegressor":
        X_train = self._fit_preprocess(X)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (self._transform(X_val), y_val)
        self._model.fit(
            X_train,
            y,
            cat_features=self.categorical_columns_,
            eval_set=eval_set,
            use_best_model=eval_set is not None,
            verbose=False,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(self._transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "CatBoostMicRegressor is a regression model and does not expose "
            "class probabilities."
        )

    def artifact_metadata(self, feature_columns: list[str]) -> dict:
        importance = self._model.get_feature_importance()
        return {
            "catboost_best_iteration": self._model.get_best_iteration(),
            "catboost_best_score": self._model.get_best_score(),
            "catboost_evals_result": self._model.get_evals_result(),
            "catboost_resolved_params": self._model.get_all_params(),
            "catboost_categorical_columns": self.categorical_columns_,
            "numeric_imputation_medians": self.numeric_medians_,
            "catboost_feature_importance": {
                column: float(score)
                for column, score in zip(feature_columns, importance, strict=False)
            },
        }

    def _fit_preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        self.feature_columns_ = X.columns.tolist()
        self.numeric_columns_ = [
            column
            for column in self.feature_columns_
            if column not in self.categorical_columns_
        ]
        self.numeric_medians_ = {}
        for column in self.numeric_columns_:
            numeric = pd.to_numeric(X[column], errors="coerce")
            median = numeric.median()
            self.numeric_medians_[column] = 0.0 if pd.isna(median) else float(median)
        return self._transform(X)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        columns = {}
        for column in self.categorical_columns_:
            if column in X.columns:
                values = X[column]
            else:
                values = pd.Series("Unknown", index=X.index)
            columns[column] = (
                values
                .fillna("Unknown")
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
            )
        for column, median in self.numeric_medians_.items():
            if column in X.columns:
                values = X[column]
            else:
                values = pd.Series(median, index=X.index)
            columns[column] = (
                pd.to_numeric(values, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(median)
                .astype(float)
            )
        prepared = pd.DataFrame(columns, index=X.index)
        return prepared.reindex(columns=self.feature_columns_)


def build_model(random_state: int = 42) -> CatBoostMicRegressor:
    """Create the CatBoost MIC regressor."""
    return CatBoostMicRegressor(random_state=random_state)


def build_tuned_model(random_state: int = 42) -> CatBoostMicRegressor:
    """Create a stronger CatBoost MIC regressor for MAE-focused validation."""
    return CatBoostMicRegressor(
        random_state=random_state,
        iterations=3500,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=20.0,
        random_strength=2.0,
        subsample=0.8,
        rsm=0.85,
        bootstrap_type="Bernoulli",
        loss_function="MAE",
        eval_metric="MAE",
        early_stopping_rounds=150,
    )


def catboost_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return feature metadata stored with the trained artifact."""
    return {
        "sequence_descriptor_columns": SequenceDescriptorEncoder(
            feature_set=DESCRIPTOR_FEATURE_SET
        ).feature_names(),
        "engineered_feature_columns": list(ENGINEERED_FEATURE_COLUMNS),
        "categorical_feature_columns": list(CATBOOST_CATEGORICAL_COLUMNS),
        "descriptor_library": "modlamp_plus_reduced_alphabet_kmers",
        "sequence_feature_set": DESCRIPTOR_FEATURE_SET,
        "target": "log10_mic",
        "duplicate_measurements": "median_log_mic_by_sequence_target",
        "null_policy": "taxonomy_unknown_numeric_train_median",
    }


__all__ = [
    "CATBOOST_CATEGORICAL_COLUMNS",
    "CatBoostMicRegressor",
    "aggregate_duplicate_measurements",
    "build_catboost_features",
    "build_engineered_physchem_features",
    "build_model",
    "build_tuned_model",
    "catboost_artifact_metadata",
    "evaluate_taxonomy_predictions",
    "load_catboost_mic_data",
]
