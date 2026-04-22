import pandas as pd
import numpy as np
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
import os
import re

INPUT_PATH = "data/processed/embeddings/sequences_with_MIC.csv"
OUTPUT_PATH = "data/processed/embeddings/features_from_sequences.csv"

CROSSCORR_WINDOW = 7

def clean_sequence(seq):
    seq = str(seq).upper().strip()
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)
    return seq

df = pd.read_csv(INPUT_PATH)
all_sequences = df.iloc[:, 0].apply(clean_sequence).tolist()
all_sequences = [s for s in all_sequences if len(s) > 0]

long_seqs  = [s for s in all_sequences if len(s) >= CROSSCORR_WINDOW]
short_seqs = [s for s in all_sequences if len(s) <  CROSSCORR_WINDOW]

print(f"Total sequences   : {len(all_sequences)}")
print(f"Long  (>= {CROSSCORR_WINDOW} AA)  : {len(long_seqs)}")
print(f"Short (<  {CROSSCORR_WINDOW} AA)  : {len(short_seqs)}")


# ── Helper ─────────────────────────────────────────────────────────────────────
def compute_features(sequences):
    """Return a DataFrame of all features for a list of sequences."""

    # 1. Global descriptors (10 features, named by modlAMP itself)
    gd = GlobalDescriptor(sequences)
    gd.calculate_all(amide=True)
    global_df = pd.DataFrame(gd.descriptor, columns=gd.featurenames)

    # 2. Eisenberg hydrophobicity + hydrophobic moment
    pd1 = PeptideDescriptor(sequences, 'eisenberg')
    pd1.calculate_global()
    pd1.calculate_moment(append=True)

    # 3. GRAVY hydrophobicity + moment (append onto pd1)
    pd1.load_scale('gravy')
    pd1.calculate_global(append=True)
    pd1.calculate_moment(append=True)

    # 4. Z3 physicochemical scale (3 descriptors via autocorr window=1)
    pd1.load_scale('z3')
    pd1.calculate_autocorr(1, append=True)

    # 5. AASI (helical AMP selectivity) global + moment
    pd1.load_scale('aasi')
    pd1.calculate_global(append=True)
    pd1.calculate_moment(append=True)

    # 6. Charge scale global + moment
    pd1.load_scale('charge_phys')
    pd1.calculate_global(append=True)
    pd1.calculate_moment(append=True)

    scale_cols = [
        'H_Eisenberg', 'uH_Eisenberg',
        'H_GRAVY',     'uH_GRAVY',
        'Z3_1',        'Z3_2',        'Z3_3',
        'H_AASI',      'uH_AASI',
        'H_Charge',    'uH_Charge',
    ]
    scale_df = pd.DataFrame(pd1.descriptor, columns=scale_cols)

    return pd.concat([global_df, scale_df], axis=1)


def compute_pepcats(sequences, window):
    """Return a DataFrame of pepcats cross-correlation features."""
    pd2 = PeptideDescriptor(sequences, 'pepcats')
    pd2.calculate_autocorr(window)
    n_feats = pd2.descriptor.shape[1]
    cols = [f"pepcats_cc_{i+1}" for i in range(n_feats)]
    return pd.DataFrame(pd2.descriptor, columns=cols)


# ── Compute for long sequences (all features) ──────────────────────────────────
long_base_df   = compute_features(long_seqs)
long_pepcats_df = compute_pepcats(long_seqs, CROSSCORR_WINDOW)
long_df = pd.concat([
    pd.Series(long_seqs, name="sequence"),
    long_base_df,
    long_pepcats_df,
], axis=1)

# ── Compute for short sequences (no pepcats → NaN) ─────────────────────────────
if short_seqs:
    short_base_df = compute_features(short_seqs)
    # Fill pepcats columns with NaN
    n_pepcats = long_pepcats_df.shape[1]
    short_pepcats_df = pd.DataFrame(
        np.nan, index=range(len(short_seqs)),
        columns=long_pepcats_df.columns
    )
    short_df = pd.concat([
        pd.Series(short_seqs, name="sequence"),
        short_base_df,
        short_pepcats_df,
    ], axis=1)
else:
    short_df = pd.DataFrame(columns=long_df.columns)

# ── Merge, restore original order, save ───────────────────────────────────────
features_df = pd.concat([long_df, short_df], ignore_index=True)

# Restore original sequence order
order = {s: i for i, s in enumerate(all_sequences)}
features_df = features_df.sort_values(
    "sequence", key=lambda col: col.map(order)
).reset_index(drop=True)

os.makedirs("data/processed/embeddings", exist_ok=True)
features_df.to_csv(OUTPUT_PATH, index=False)

print("\nDone!")
print(f"Sequences processed : {len(features_df)}")
print(f"Feature shape       : {features_df.shape}")
print(f"\nColumns:\n{list(features_df.columns)}")