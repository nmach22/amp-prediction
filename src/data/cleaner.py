"""
Sequence cleaning and filtering utilities.

Rules applied (all configurable via keyword args):
  - Remove sequences with non-standard amino acids (default: True)
  - Remove duplicate sequences (default: True)
  - Filter by minimum / maximum length (default: 5–200 AA)
"""

import re
import pandas as pd

# IUPAC standard amino acid alphabet
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequences(
    df: pd.DataFrame,
    min_len: int = 5,
    max_len: int = 200,
    remove_nonstandard: bool = True,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """Apply cleaning rules to a raw sequence DataFrame.

    Args:
        df: DataFrame with a 'sequence' column.
        min_len: Minimum sequence length to keep.
        max_len: Maximum sequence length to keep.
        remove_nonstandard: Drop sequences with non-standard amino acids.
        remove_duplicates: Keep only the first occurrence of each sequence.

    Returns:
        Cleaned DataFrame with reset index.
    """
    df = df.copy()
    df["sequence"] = df["sequence"].str.upper().str.strip()

    # Length filter
    lengths = df["sequence"].str.len()
    df = df[lengths.between(min_len, max_len)]

    # Non-standard amino acid filter
    if remove_nonstandard:
        pattern = re.compile(f"[^{''.join(STANDARD_AA)}]")
        mask = df["sequence"].apply(lambda s: not bool(pattern.search(s)))
        df = df[mask]

    # Deduplicate
    if remove_duplicates:
        df = df.drop_duplicates(subset="sequence", keep="first")

    return df.reset_index(drop=True)

