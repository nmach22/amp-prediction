"""MIC regression baseline using target-species taxonomy features."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.mic_baseline import (
    MicBaselineRegressor,
    build_model,
    encode_sequences,
    evaluate_predictions,
    infer_test_csv,
    split_train_val_by_sequence,
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


def train_and_evaluate(
    input_csv: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    test_csv: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Train taxonomy MIC baseline and write predictions, metrics, and model."""
    output_path = Path(output_dir)
    tables_dir = output_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_taxonomy_mic_data(input_csv)
    splits = split_train_val_by_sequence(df, random_state=random_state)
    resolved_test_csv = (
        Path(test_csv) if test_csv is not None else infer_test_csv(input_csv)
    )
    test_df = (
        load_taxonomy_mic_data(resolved_test_csv)
        if resolved_test_csv is not None
        else splits.test
    )

    X_train = build_taxonomy_features(splits.train)
    y_train = splits.train["log_mic"].to_numpy()
    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)

    metrics_by_split: dict[str, dict[str, float]] = {}
    prediction_frames = []
    prediction_columns = [
        column
        for column in [
            "sequence",
            "target_activity_name",
            "activity",
            "log_mic",
            "Phylum",
            "Class",
            "Order",
            "Family",
            "Genus",
        ]
        if column in df.columns
    ]

    for split_name, split_df in [
        ("train", splits.train),
        ("val", splits.val),
        ("test", test_df),
    ]:
        if split_df.empty:
            continue
        X = build_taxonomy_features(split_df).reindex(
            columns=X_train.columns, fill_value=0.0
        )
        y_true = split_df["log_mic"].to_numpy()
        y_pred = model.predict(X)
        metrics_by_split[split_name] = evaluate_taxonomy_predictions(
            split_df, y_true, y_pred
        )

        if split_name in {"val", "test"}:
            pred_df = split_df[prediction_columns].copy()
            pred_df["split"] = split_name
            pred_df["pred_log_mic"] = y_pred
            pred_df["pred_mic"] = np.power(10.0, y_pred)
            prediction_frames.append(pred_df)

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        tables_dir / "taxonomy_mic_baseline_predictions.csv", index=False
    )
    pd.DataFrame(metrics_by_split).T.to_csv(
        tables_dir / "taxonomy_mic_baseline_metrics.csv", index_label="split"
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": X_train.columns.tolist(),
            "taxonomy_feature_columns": taxonomy_feature_columns(df),
        },
        output_path / "taxonomy_mic_baseline_model.joblib",
    )
    return metrics_by_split


__all__ = [
    "MicBaselineRegressor",
    "build_taxonomy_features",
    "load_taxonomy_mic_data",
    "taxonomy_feature_columns",
    "train_and_evaluate",
]
