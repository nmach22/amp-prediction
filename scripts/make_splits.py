"""Create fixed train/test splits for MIC experiments.

The test split is held out for final model selection. Individual model
pipelines should read train.csv and make their own train/validation split.

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --input data/processed/amp_mic_activities_taxonomy_features.csv

Output files:
    data/processed/splits/train.csv
    data/processed/splits/test.csv
"""

from pathlib import Path
import argparse
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).parents[1]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger(__name__)
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
NONSTANDARD_PATTERN = re.compile(f"[^{STANDARD_AA}]")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create fixed train/test MIC splits.")
    p.add_argument(
        "--input",
        default="data/processed/amp_mic_activities_taxonomy_features.csv",
        help="Input MIC CSV, usually the taxonomy-feature dataset.",
    )
    p.add_argument("--train", type=float, default=0.85, help="Train fraction")
    p.add_argument("--test", type=float, default=0.15, help="Held-out test fraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def _resolve_input(path: str) -> Path:
    input_path = Path(path)
    if input_path.exists():
        return input_path
    raw_path = ROOT / "data" / "raw" / path
    if raw_path.exists():
        return raw_path
    processed_path = ROOT / "data" / "processed" / path
    if processed_path.exists():
        return processed_path
    raise FileNotFoundError(f"Could not find input CSV: {path}")


def clean_mic_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"sequence", "activity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["sequence", "activity"])
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.upper().str.strip()
    cleaned["activity"] = pd.to_numeric(cleaned["activity"], errors="coerce")
    cleaned = cleaned.dropna(subset=["activity"])
    cleaned = cleaned[cleaned["activity"] > 0]
    cleaned = cleaned[cleaned["sequence"].str.len() > 0]
    cleaned = cleaned[~cleaned["sequence"].str.contains(NONSTANDARD_PATTERN)]
    if "target_is_bacteria" in cleaned.columns:
        cleaned = cleaned[cleaned["target_is_bacteria"].fillna(0).astype(int) == 1]
    if "gram_status" in cleaned.columns:
        cleaned = cleaned[
            cleaned["gram_status"].isin(["gram_positive", "gram_negative"])
        ]
    return cleaned.reset_index(drop=True)


def split_train_test_by_sequence(
    df: pd.DataFrame,
    train_size: float = 0.85,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_size + test_size, 1.0):
        raise ValueError("train and test fractions must sum to 1.0")

    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["sequence"]))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    input_path = _resolve_input(args.input)
    log.info("Loading data: %s", input_path)
    df = pd.read_csv(input_path)
    log.info("Raw rows: %s", len(df))

    df = clean_mic_rows(df)
    log.info("After cleaning: %s rows", len(df))
    log.info("Unique sequences: %s", df["sequence"].nunique())

    df_train, df_test = split_train_test_by_sequence(
        df,
        train_size=args.train,
        test_size=args.test,
        random_state=args.seed,
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for stale_name in ["val.csv"]:
        stale_path = SPLITS_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()

    for name, split_df in [("train", df_train), ("test", df_test)]:
        out = SPLITS_DIR / f"{name}.csv"
        split_df.to_csv(out, index=False)
        log.info("  %-5s: %7d rows -> %s", name, len(split_df), out)
        if "gram_status" in split_df.columns:
            log.info("\n%s", split_df["gram_status"].value_counts().to_string())

    log.info("Splits saved. Keep test.csv untouched until final model evaluation.")


if __name__ == "__main__":
    main()
