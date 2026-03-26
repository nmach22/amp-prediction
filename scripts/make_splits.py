"""
scripts/make_splits.py
======================
One-time script that builds fixed train / val / test splits from the
cleaned DBAASP data and saves them to data/processed/splits/.

Both collaborators commit these CSVs so every experiment uses exactly
the same partitions.

Usage:
    python scripts/make_splits.py --input dbaasp_export.csv \
                                  --train 0.70 \
                                  --val   0.15 \
                                  --test  0.15 \
                                  --seed  42

Output files (written to data/processed/splits/):
    train.csv   val.csv   test.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ── project root is two levels up from this script ──────────────────────────
ROOT = Path(__file__).parents[1]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

# ── make src importable when running the script directly ────────────────────
import sys
sys.path.insert(0, str(ROOT))

from src.data import load_raw, clean_sequences
from src.utils import get_logger, set_seed

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create fixed train/val/test splits.")
    p.add_argument(
        "--input",
        required=True,
        help="Filename inside data/raw/ (e.g. dbaasp_export.csv)",
    )
    p.add_argument("--train", type=float, default=0.70, help="Train fraction")
    p.add_argument("--val",   type=float, default=0.15, help="Val fraction")
    p.add_argument("--test",  type=float, default=0.15, help="Test fraction")
    p.add_argument("--seed",  type=int,   default=42,   help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise ValueError("train + val + test fractions must sum to 1.0")

    set_seed(args.seed)

    # ── Load & clean ─────────────────────────────────────────────────────────
    log.info(f"Loading raw data: {args.input}")
    df = load_raw(args.input)
    log.info(f"Raw samples: {len(df)}")

    df = clean_sequences(df)
    log.info(f"After cleaning: {len(df)} samples")

    # ── Stratified splits ────────────────────────────────────────────────────
    # Step 1: separate test set
    test_frac_of_full = args.test
    val_frac_of_remainder = args.val / (args.train + args.val)

    df_trainval, df_test = train_test_split(
        df,
        test_size=test_frac_of_full,
        stratify=df["activity"],
        random_state=args.seed,
    )
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_frac_of_remainder,
        stratify=df_trainval["activity"],
        random_state=args.seed,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        out = SPLITS_DIR / f"{name}.csv"
        split_df.to_csv(out, index=False)
        pos = split_df["activity"].sum()
        log.info(
            f"  {name:5s}: {len(split_df):5d} samples "
            f"({pos} AMP / {len(split_df) - pos} non-AMP) → {out}"
        )

    log.info("Splits saved. Commit data/processed/splits/ to git.")


if __name__ == "__main__":
    main()

