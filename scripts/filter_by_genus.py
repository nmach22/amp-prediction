"""Filter AMP MIC activities by bacterial genus.

Extracts the genus (first word of target_activity_name), removes fungi,
and keeps only genera with at least --min-count data points.

Usage:
    python scripts/filter_by_genus.py
    python scripts/filter_by_genus.py --min-count 1000
    python scripts/filter_by_genus.py --input data/raw/amp_mic_activities.csv \
                                      --output data/processed/amp_mic_filtered.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FUNGI_GENERA = [
    "Candida", "Cryptococcus", "Saccharomyces", "Trichosporon",
    "Aspergillus", "Fusarium", "Penicillium", "Botrytis", "Trichophyton",
]

DEFAULT_MIN_COUNT = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter AMP MIC activities by bacterial genus.",
    )
    parser.add_argument(
        "--input",
        default="data/raw/amp_mic_activities.csv",
        help="Input CSV with target_activity_name column.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/amp_mic_filtered.csv",
        help="Output CSV with genus filtering applied.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
        help=f"Minimum samples per genus to keep (default: {DEFAULT_MIN_COUNT}).",
    )
    return parser.parse_args()


def extract_genus(series: pd.Series) -> pd.Series:
    """Extract genus (first word) from target_activity_name, title-cased."""
    return (
        series
        .astype(str)
        .str.strip()
        .str.split()
        .str[0]
        .str.title()
        .where(series.notna(), other=pd.NA)
    )


def filter_by_genus(
    df: pd.DataFrame,
    min_count: int = DEFAULT_MIN_COUNT,
    fungi: list[str] | None = None,
) -> pd.DataFrame:
    """Add genus_label, remove fungi, and keep genera with >= min_count rows."""
    if fungi is None:
        fungi = FUNGI_GENERA

    df = df.copy()
    df["genus_label"] = extract_genus(df["target_activity_name"])

    df = df[~df["genus_label"].isin(fungi)]

    counts = df["genus_label"].value_counts()
    keep = counts[counts >= min_count].index
    df = df[df["genus_label"].isin(keep)].reset_index(drop=True)

    return df


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    print(f"Input:  {len(df)} rows")

    result = filter_by_genus(df, min_count=args.min_count)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    genera = result["genus_label"].value_counts()
    print(f"Output: {len(result)} rows, {len(genera)} genera (min_count={args.min_count})")
    print()
    print(genera.to_string())


if __name__ == "__main__":
    main()
