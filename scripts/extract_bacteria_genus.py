"""Extract bacteria genus+species from target_activity_name.

Takes the first two words of target_activity_name as a new
``bacteria_genus`` column (e.g. "Staphylococcus aureus").

Usage:
    python scripts/extract_bacteria_genus.py
    python scripts/extract_bacteria_genus.py --input data/raw/amp_mic_activities.csv \
                                              --output data/interim/amp_mic_with_genus.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def extract_bacteria_genus(series: pd.Series) -> pd.Series:
    """Return first two whitespace-delimited words from each value."""
    return (
        series
        .astype(str)
        .str.strip()
        .str.split()
        .str[:2]
        .str.join(" ")
        .where(series.notna(), other=pd.NA)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract bacteria genus+species from target_activity_name.",
    )
    parser.add_argument(
        "--input",
        default="data/raw/amp_mic_activities.csv",
        help="Input CSV with target_activity_name column.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/amp_mic_with_genus.csv",
        help="Output CSV with added bacteria_genus column.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["bacteria_genus"] = extract_bacteria_genus(df["target_activity_name"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    n_unique = df["bacteria_genus"].dropna().nunique()
    print(f"Saved {len(df)} rows to {args.output}  ({n_unique} unique bacteria genus values)")


if __name__ == "__main__":
    main()
