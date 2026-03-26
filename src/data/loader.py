"""
Loaders for raw DBAASP exports and pre-saved split CSVs.

Expected raw CSV columns (from DBAASP):
    sequence  – amino acid sequence (str)
    activity  – binary label: 1 = AMP, 0 = non-AMP
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
SPLITS_DIR = Path(__file__).parents[2] / "data" / "processed" / "splits"


def load_raw(filename: str) -> pd.DataFrame:
    """Load a raw DBAASP CSV export.

    Args:
        filename: name of the CSV file inside data/raw/

    Returns:
        DataFrame with at least 'sequence' and 'activity' columns.
    """
    path = RAW_DIR / filename
    df = pd.read_csv(path)
    _validate_columns(df)
    return df


def load_split(split: str) -> pd.DataFrame:
    """Load a pre-saved data split.

    Args:
        split: one of 'train', 'val', 'test'

    Returns:
        DataFrame with 'sequence' and 'activity' columns.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val', or 'test', got: {split!r}")
    path = SPLITS_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/make_splits.py first."
        )
    return pd.read_csv(path)


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"sequence", "activity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

