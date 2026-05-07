"""
DBAASP Dataset - Step 1: Train / Validation / Test Split
=========================================================
Split ratio  : 70 / 15 / 15
Activity type: Continuous MIC values (log-transformed before stratification)
Stratify on  : gram_status  +  binned log-MIC  (combined key)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# ── 0. Config ────────────────────────────────────────────────────────────────
INPUT_CSV   = "./amp_mic_activities_gram_classified.csv"
OUTPUT_DIR  = Path("splits")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
N_MIC_BINS  = 5                        # number of quantile bins for stratification

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df):,} rows  |  columns: {list(df.columns)}")

# Basic sanity checks
assert {"sequence", "gram_status", "activity"}.issubset(df.columns), \
    "Missing expected columns!"

# Drop rows with nulls in key columns
before = len(df)
df = df.dropna(subset=["sequence", "gram_status", "activity"])
print(f"Dropped {before - len(df)} rows with nulls → {len(df):,} rows remain")

# ── 2. Log-transform MIC for better distribution ──────────────────────────────
# MIC values are typically right-skewed; log10 makes them more Gaussian
df["log_activity"] = np.log10(df["activity"].clip(lower=1e-6))

# ── 3. Build a stratification key: gram_status + binned log-MIC ──────────────
df["mic_bin"] = pd.qcut(
    df["log_activity"],
    q=N_MIC_BINS,
    labels=False,
    duplicates="drop"
)
df["strat_key"] = df["gram_status"].astype(str) + "_bin" + df["mic_bin"].astype(str)

# Collapse rare strat keys (< 2 samples) to avoid split errors
key_counts = df["strat_key"].value_counts()
rare_keys  = key_counts[key_counts < 2].index
df.loc[df["strat_key"].isin(rare_keys), "strat_key"] = "rare"
print(f"Stratification keys: {df['strat_key'].nunique()} unique groups")

# ── 4. Split: 70 train  |  15 val  |  15 test ────────────────────────────────
# First cut: 70% train, 30% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=df["strat_key"]
)

# Second cut: split temp 50/50 → 15% val, 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_df["strat_key"]
)

# ── 5. Drop helper columns before saving ─────────────────────────────────────
KEEP_COLS = ["sequence", "gram_status", "log_activity"]

train_df = train_df[KEEP_COLS].reset_index(drop=True)
val_df   = val_df[KEEP_COLS].reset_index(drop=True)
test_df  = test_df[KEEP_COLS].reset_index(drop=True)

# ── 6. Save ───────────────────────────────────────────────────────────────────
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR  / "val.csv",   index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv",  index=False)

# ── 7. Summary report ────────────────────────────────────────────────────────
print("\n── Split Summary ──────────────────────────────────────────────────────")
for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    pct = len(split) / len(df) * 100
    print(f"\n{name:6s}  {len(split):>6,} rows ({pct:.1f}%)")
    print(f"  gram_status distribution:")
    print(split["gram_status"].value_counts().to_string(header=False)
          .replace("\n", "\n    "))

print("\n✓ Files saved to:", OUTPUT_DIR.resolve())