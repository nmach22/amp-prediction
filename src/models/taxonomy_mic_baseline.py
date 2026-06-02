"""MIC regression baseline using target-species taxonomy features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.mic_baseline import (
    MicBaselineRegressor,
    encode_sequences,
    evaluate_predictions,
)

TAXONOMY_FEATURE_PREFIXES = ("Phylum_", "Class_", "Order_", "Family_", "Genus_")
TAXONOMY_METADATA_COLUMNS = (
    "target_activity_name",
    "Phylum",
    "Class",
    "Order",
    "Family",
    "Genus",
    "gram_status",
    "is_bacteria",
    "taxid",
    "taxonomy_query_name",
    "taxonomy_matched_name",
)


def taxonomy_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return one-hot taxonomy feature columns from an enriched MIC dataframe."""
    columns = [
        column
        for column in df.columns
        if column.startswith(TAXONOMY_FEATURE_PREFIXES)
    ]
    if "target_is_bacteria" in df.columns:
        columns.append("target_is_bacteria")
    if not columns:
        raise ValueError(
            "No taxonomy feature columns found. Expected columns starting with "
            "Phylum_, Class_, Order_, Family_, or Genus_."
        )
    return columns


def load_taxonomy_mic_data(path: str | Path) -> pd.DataFrame:
    """Load and clean MIC rows with target-species taxonomy features."""
    df = pd.read_csv(path)
    required = {"sequence", "target_activity_name", "activity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feature_columns = taxonomy_feature_columns(df)
    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["sequence", "target_activity_name", "activity"])
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.upper().str.strip()
    cleaned["target_activity_name"] = cleaned["target_activity_name"].astype(str).str.strip()
    cleaned["activity"] = pd.to_numeric(cleaned["activity"], errors="coerce")
    cleaned = cleaned.dropna(subset=["activity"])
    cleaned = cleaned[cleaned["activity"] > 0]
    cleaned = cleaned[cleaned["sequence"].str.len() > 0]

    from src.models.mic_baseline import NONSTANDARD_PATTERN

    cleaned = cleaned[~cleaned["sequence"].str.contains(NONSTANDARD_PATTERN)]
    if "target_is_bacteria" in cleaned.columns:
        cleaned = cleaned[cleaned["target_is_bacteria"].fillna(0).astype(int) == 1]

    for column in feature_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0.0)

    cleaned = cleaned.copy()
    cleaned["log_mic"] = np.log10(cleaned["activity"])

    keep_columns = list(dict.fromkeys([
        "sequence",
        "target_activity_name",
        "activity",
        "log_mic",
        *[column for column in TAXONOMY_METADATA_COLUMNS if column in cleaned.columns],
        *feature_columns,
    ]))
    return cleaned[keep_columns].reset_index(drop=True)


def build_taxonomy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features from peptide sequence and target taxonomy columns."""
    sequence_features = encode_sequences(df["sequence"])
    taxonomy_features = df[taxonomy_feature_columns(df)].astype(float)
    return pd.concat(
        [
            sequence_features.reset_index(drop=True),
            taxonomy_features.reset_index(drop=True),
        ],
        axis=1,
    )


def evaluate_taxonomy_predictions(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute overall metrics plus broad taxonomy-group diagnostics."""
    metrics = evaluate_predictions(_with_gram_status(df), y_true, y_pred)

    if "Phylum" in df.columns:
        for phylum, group in df.groupby("Phylum", dropna=False):
            if len(group) < 100:
                continue
            mask = df.index.isin(group.index)
            safe_name = str(phylum).lower().replace(" ", "_")
            metrics[f"phylum_{safe_name}_mae"] = float(
                np.mean(np.abs(y_true[mask] - y_pred[mask]))
            )
    return metrics


def _with_gram_status(df: pd.DataFrame) -> pd.DataFrame:
    """Provide a compatible gram_status column for shared MIC metrics."""
    if "gram_status" in df.columns:
        return df
    compatible = df.copy()
    compatible["gram_status"] = "unknown"
    return compatible


__all__ = [
    "MicBaselineRegressor",
    "build_taxonomy_features",
    "evaluate_taxonomy_predictions",
    "load_taxonomy_mic_data",
    "taxonomy_feature_columns",
]
