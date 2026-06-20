"""Summarize MIC prediction errors by target and sequence diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = "results/tables/catboost_mic_tuned_predictions.csv"
DEFAULT_SOURCE = "data/processed/amp_mic_activities_taxonomy_features.csv"
DEFAULT_OUTPUT = "results/tables/catboost_mic_tuned_error_analysis.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write grouped MIC error diagnostics from prediction outputs."
    )
    parser.add_argument(
        "--predictions",
        default=DEFAULT_PREDICTIONS,
        help="Predictions CSV with log_mic and pred_log_mic columns.",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Optional source CSV used to compute duplicate sequence-target density.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV for grouped error analysis.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=30,
        help="Minimum rows required for a non-overall segment.",
    )
    return parser.parse_args()


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load predictions and attach row-level error columns."""
    df = pd.read_csv(path)
    required = {"log_mic", "pred_log_mic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")
    df = df.copy()
    df["error"] = df["pred_log_mic"] - df["log_mic"]
    df["abs_error"] = df["error"].abs()
    df["sequence_length"] = df["sequence"].astype(str).str.len()
    df["length_bin"] = pd.cut(
        df["sequence_length"],
        bins=[0, 10, 20, 40, np.inf],
        labels=["<=10", "11-20", "21-40", ">40"],
        include_lowest=True,
    ).astype(str)
    df["log_mic_bin"] = _safe_qcut(df["log_mic"], q=4, prefix="log_mic_q")
    return df


def attach_duplicate_counts(
    predictions: pd.DataFrame,
    source_path: str | Path | None,
) -> pd.DataFrame:
    """Add source duplicate counts for sequence-target pairs when available."""
    if not source_path or not Path(source_path).exists():
        predictions = predictions.copy()
        predictions["duplicate_count"] = 1
        predictions["duplicate_count_bin"] = "1"
        return predictions

    source = pd.read_csv(source_path, usecols=["sequence", "target_activity_name"])
    source["sequence"] = source["sequence"].astype(str).str.upper().str.strip()
    source["target_activity_name"] = source["target_activity_name"].astype(str).str.strip()
    counts = (
        source.groupby(["sequence", "target_activity_name"], dropna=False)
        .size()
        .rename("duplicate_count")
        .reset_index()
    )
    enriched = predictions.copy()
    enriched["sequence"] = enriched["sequence"].astype(str).str.upper().str.strip()
    enriched["target_activity_name"] = (
        enriched["target_activity_name"].astype(str).str.strip()
    )
    enriched = enriched.merge(
        counts,
        on=["sequence", "target_activity_name"],
        how="left",
    )
    enriched["duplicate_count"] = enriched["duplicate_count"].fillna(1).astype(int)
    enriched["duplicate_count_bin"] = pd.cut(
        enriched["duplicate_count"],
        bins=[0, 1, 2, 5, 10, np.inf],
        labels=["1", "2", "3-5", "6-10", ">10"],
        include_lowest=True,
    ).astype(str)
    return enriched


def build_error_analysis(df: pd.DataFrame, min_count: int = 30) -> pd.DataFrame:
    """Return grouped error summaries for common MIC diagnostics."""
    groups = [
        ("overall", None),
        ("gram_status", "gram_status"),
        ("phylum", "Phylum"),
        ("genus", "Genus"),
        ("length_bin", "length_bin"),
        ("log_mic_bin", "log_mic_bin"),
        ("duplicate_count_bin", "duplicate_count_bin"),
    ]
    rows = []
    for segment_type, column in groups:
        if column is None:
            rows.append(_summary_row(df, segment_type, "all"))
            continue
        if column not in df.columns:
            continue
        for value, group in df.groupby(column, dropna=False):
            if len(group) < min_count:
                continue
            rows.append(_summary_row(group, segment_type, str(value)))
    return pd.DataFrame(rows).sort_values(
        ["segment_type", "mae"],
        ascending=[True, False],
        ignore_index=True,
    )


def write_error_analysis(
    predictions_path: str | Path,
    output_path: str | Path,
    source_path: str | Path | None = None,
    min_count: int = 30,
) -> pd.DataFrame:
    """Load inputs, compute grouped errors, and write the output CSV."""
    predictions = load_predictions(predictions_path)
    enriched = attach_duplicate_counts(predictions, source_path)
    analysis = build_error_analysis(enriched, min_count=min_count)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output, index=False)
    return analysis


def _summary_row(df: pd.DataFrame, segment_type: str, segment: str) -> dict:
    abs_error = df["abs_error"].to_numpy()
    error = df["error"].to_numpy()
    log_error = abs_error
    return {
        "segment_type": segment_type,
        "segment": segment,
        "n": int(len(df)),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "median_ae": float(np.median(abs_error)),
        "mean_error": float(error.mean()),
        "within_2fold": float(np.mean(log_error <= np.log10(2.0))),
        "within_4fold": float(np.mean(log_error <= np.log10(4.0))),
    }


def _safe_qcut(values: pd.Series, q: int, prefix: str) -> pd.Series:
    try:
        bins = pd.qcut(values, q=q, duplicates="drop")
    except ValueError:
        return pd.Series(f"{prefix}_all", index=values.index)
    labels = []
    for interval in bins.cat.categories:
        labels.append(f"{prefix}_{interval.left:.2f}_{interval.right:.2f}")
    return bins.cat.rename_categories(labels).astype(str)


def main() -> None:
    args = parse_args()
    analysis = write_error_analysis(
        predictions_path=args.predictions,
        source_path=args.source,
        output_path=args.output,
        min_count=args.min_count,
    )
    print(analysis.to_string(index=False))


if __name__ == "__main__":
    main()
