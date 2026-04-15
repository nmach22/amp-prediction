import pandas as pd
import numpy as np
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor
import os
import re

INPUT_PATH = "data/processed/embeddings/sequences_with_MIC.csv"
OUTPUT_PATH = "data/processed/embeddings/features_from_sequences.csv"

def clean_sequence(seq):
    seq = str(seq).upper().strip()
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)
    return seq

df = pd.read_csv(INPUT_PATH)
all_sequences = df.iloc[:, 0].apply(clean_sequence).tolist()

CROSSCORR_WINDOW = 7

# Filter: must be non-empty AND long enough for crosscorr
sequences = [s for s in all_sequences if len(s) >= CROSSCORR_WINDOW]
short_sequences = [s for s in all_sequences if 0 < len(s) < CROSSCORR_WINDOW]

if short_sequences:
    print(f"Skipping {len(short_sequences)} sequences shorter than window {CROSSCORR_WINDOW}: {short_sequences}")

print(f"Valid sequences: {len(sequences)}")

# ── 1. GlobalDescriptor ────────────────────────────────────────────────────────
gd = GlobalDescriptor(sequences)
gd.calculate_all(amide=True)
global_df = pd.DataFrame(gd.descriptor, columns=gd.featurenames)

# ── 2. PeptideDescriptor: multiple AA scales ───────────────────────────────────
pd1 = PeptideDescriptor(sequences, 'eisenberg')
pd1.calculate_global()
pd1.calculate_moment(append=True)

pd1.load_scale('gravy')
pd1.calculate_global(append=True)
pd1.calculate_moment(append=True)

pd1.load_scale('z3')
pd1.calculate_autocorr(1, append=True)

pd2 = PeptideDescriptor(sequences, 'pepcats')
pd2.calculate_crosscorr(CROSSCORR_WINDOW)

pd3 = PeptideDescriptor(sequences, 'aasi')
pd3.calculate_global()
pd3.calculate_moment(append=True)

pd4 = PeptideDescriptor(sequences, 'charge_phys')
pd4.calculate_global()
pd4.calculate_moment(append=True)

# ── 3. Assemble ────────────────────────────────────────────────────────────────
def make_df(descriptor_array, prefix):
    arr = np.array(descriptor_array)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, columns=[f"{prefix}_{i}" for i in range(arr.shape[1])])

features_df = pd.concat([
    pd.Series(sequences, name="sequence"),
    global_df.reset_index(drop=True),
    make_df(pd1.descriptor, "scale"),
    make_df(pd2.descriptor, "pepcats"),
    make_df(pd3.descriptor, "aasi"),
    make_df(pd4.descriptor, "charge"),
], axis=1)

os.makedirs("data/processed/embeddings", exist_ok=True)
features_df.to_csv(OUTPUT_PATH, index=False)
print("Done.")
print("Sequences processed:", len(sequences))
print("Feature shape:", features_df.shape)