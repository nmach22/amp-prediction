"""Per-genus XGBoost MIC regression: train a separate model per genus_label."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.sequence_descriptors import SequenceDescriptorEncoder
from src.models.mic_baseline import GRAM_CLASSES, evaluate_predictions
from src.models.xgboost_mic import (
    XGBoostMicRegressor,
    aggregate_duplicate_measurements,
    build_model as build_xgboost_model,
)

GENUS_GROUPS = ("Staphylococcus", "Escherichia", "Pseudomonas", "Bacillus", "Klebsiella")


def load_per_genus_mic_data(path: str | Path) -> pd.DataFrame:
    """Load taxonomy-enriched MIC data, keeping genus_label for grouping."""
    df = pd.read_csv(path)
    required = {"sequence", "target_activity_name", "activity", "genus_label", "gram_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    from src.models.mic_baseline import NONSTANDARD_PATTERN

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["sequence", "target_activity_name", "activity"])
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.upper().str.strip()
    cleaned["target_activity_name"] = cleaned["target_activity_name"].astype(str).str.strip()
    cleaned["activity"] = pd.to_numeric(cleaned["activity"], errors="coerce")
    cleaned = cleaned.dropna(subset=["activity"])
    cleaned = cleaned[cleaned["activity"] > 0]
    cleaned = cleaned[cleaned["sequence"].str.len() > 0]
    cleaned = cleaned[~cleaned["sequence"].str.contains(NONSTANDARD_PATTERN)]
    if "target_is_bacteria" in cleaned.columns:
        cleaned = cleaned[cleaned["target_is_bacteria"].fillna(0).astype(int) == 1]

    cleaned["log_mic"] = np.log10(cleaned["activity"])
    cleaned = cleaned[cleaned["gram_status"].isin(GRAM_CLASSES)].reset_index(drop=True)
    cleaned = cleaned[cleaned["genus_label"].isin(GENUS_GROUPS)].reset_index(drop=True)

    keep_columns = list(dict.fromkeys([
        "sequence", "target_activity_name", "activity", "log_mic",
        "genus_label", "gram_status",
    ]))
    return aggregate_duplicate_measurements(cleaned[keep_columns].reset_index(drop=True))


def build_per_genus_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build sequence descriptor + gram features (no taxonomy one-hot columns)."""
    descriptor_encoder = SequenceDescriptorEncoder(feature_set="full_modlamp")
    sequence_features = descriptor_encoder.encode(df["sequence"])
    gram_features = pd.get_dummies(df["gram_status"], prefix="gram", dtype=float)
    for column in ["gram_gram_negative", "gram_gram_positive"]:
        if column not in gram_features:
            gram_features[column] = 0.0
    gram_features = gram_features[["gram_gram_negative", "gram_gram_positive"]]
    return pd.concat(
        [
            sequence_features.reset_index(drop=True),
            gram_features.reset_index(drop=True),
        ],
        axis=1,
    )


def evaluate_per_genus_predictions(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute regression metrics for a single-genus subset."""
    compatible = df.copy()
    if "gram_status" not in compatible.columns:
        compatible["gram_status"] = "unknown"
    return evaluate_predictions(compatible, y_true, y_pred)


def per_genus_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return feature metadata stored with the trained artifact."""
    return {
        "sequence_descriptor_columns": SequenceDescriptorEncoder(
            feature_set="full_modlamp"
        ).feature_names(),
        "descriptor_library": "modlamp",
        "sequence_feature_set": "full_modlamp",
        "target": "log10_mic",
        "genus_groups": list(GENUS_GROUPS),
    }


__all__ = [
    "GENUS_GROUPS",
    "build_per_genus_features",
    "build_xgboost_model",
    "evaluate_per_genus_predictions",
    "load_per_genus_mic_data",
    "per_genus_artifact_metadata",
]
