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
# Filter out empty strings but keep short sequences
all_sequences = df.iloc[:, 0].apply(clean_sequence).tolist()
sequences = [s for s in all_sequences if len(s) > 0]

print(f"Processing {len(sequences)} sequences.")

# 1. GlobalDescriptor (Names are built-in)
gd = GlobalDescriptor(sequences)
gd.calculate_all(amide=True)
global_df = pd.DataFrame(gd.descriptor, columns=gd.featurenames)


# 2. PeptideDescriptor: Processing scales and getting names
def get_descriptor_df(seqs, scale, mode='global', window=None):
    desc = PeptideDescriptor(seqs, scale)
    if mode == 'global_moment':
        desc.calculate_global()
        desc.calculate_moment(append=True)
    elif mode == 'autocorr':
        desc.calculate_autocorr(1)
    elif mode == 'crosscorr':
        # To handle short sequences, we cap the window at the length of the shortest sequence
        min_len = min([len(s) for s in seqs])
        actual_window = min(window, min_len) if window else min_len
        desc.calculate_crosscorr(actual_window)

    # Use desc.featurenames for appropriate column descriptions
    return pd.DataFrame(desc.descriptor, columns=[f"{scale}_{name}" for name in desc.featurenames])


# Eisenberg & Gravy
pd1 = PeptideDescriptor(sequences, 'eisenberg')
pd1.calculate_global()
pd1.calculate_moment(append=True)
pd1.load_scale('gravy')
pd1.calculate_global(append=True)
pd1.calculate_moment(append=True)
# Naming for pd1 is a mix, so we manually label or use the internal featurenames
pd1_cols = [f"eisenberg_{n}" for n in pd1.featurenames[:2]] + [f"gravy_{n}" for n in pd1.featurenames[2:]]
df_pd1 = pd.DataFrame(pd1.descriptor, columns=pd1_cols)

# Z3 Autocorr
pd_z3 = PeptideDescriptor(sequences, 'z3')
pd_z3.calculate_autocorr(1)
df_z3 = pd.DataFrame(pd_z3.descriptor, columns=[f"z3_{n}" for n in pd_z3.featurenames])

# Pepcats Crosscorr (The tricky one for short sequences)
# We set window to 1 if we want to include very short peptides safely
pd_pepcats = PeptideDescriptor(sequences, 'pepcats')
pd_pepcats.calculate_crosscorr(window=1)
df_pepcats = pd.DataFrame(pd_pepcats.descriptor, columns=[f"pepcats_{n}" for n in pd_pepcats.featurenames])

# AASI
pd_aasi = PeptideDescriptor(sequences, 'aasi')
pd_aasi.calculate_global()
pd_aasi.calculate_moment(append=True)
df_aasi = pd.DataFrame(pd_aasi.descriptor, columns=[f"aasi_{n}" for n in pd_aasi.featurenames])

# Charge Phys
pd_charge = PeptideDescriptor(sequences, 'charge_phys')
pd_charge.calculate_global()
pd_charge.calculate_moment(append=True)
df_charge = pd.DataFrame(pd_charge.descriptor, columns=[f"charge_{n}" for n in pd_charge.featurenames])

# 3. Assemble
features_df = pd.concat([
    pd.Series(sequences, name="sequence"),
    global_df.reset_index(drop=True),
    df_pd1.reset_index(drop=True),
    df_z3.reset_index(drop=True),
    df_pepcats.reset_index(drop=True),
    df_aasi.reset_index(drop=True),
    df_charge.reset_index(drop=True)
], axis=1)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
features_df.to_csv(OUTPUT_PATH, index=False)

print("Done.")
print(f"Sequences processed: {len(sequences)}")
print(f"Feature shape: {features_df.shape}")