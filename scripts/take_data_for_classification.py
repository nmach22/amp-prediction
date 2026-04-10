import pandas as pd
import re
import os

INPUT_PATH = "data/raw/dbaasp_full.csv"
OUTPUT_PATH = "data/processed/embeddings/data_for_classification.csv"

ACTIVE_THRESHOLD = 25 


# --- Approx molecular weight ---
def compute_mw(sequence):
    return len(sequence) * 110


# --- Convert µM to µg/ml ---
def convert_uM_to_ugml(value, sequence):
    mw = compute_mw(sequence)
    return value * mw / 1000


# --- Check if numeric ---
def is_numeric(x):
    try:
        float(x)
        return True
    except:
        return False


# --- Normalize unit ---
def normalize_unit(unit):
    unit = str(unit).strip()
    unit = unit.replace("μ", "µ") 
    return unit


# --- Main ---
df = pd.read_csv(INPUT_PATH)

# Keep only MIC rows
df = df[df.iloc[:, 4] == "MIC"].copy()

clean_rows = []
unknown_rows = []

for _, row in df.iterrows():
    try:
        sequence = row.iloc[3]
        species = row.iloc[4 - 1]  
        activity_value = row.iloc[5]
        concentration = str(row.iloc[6])
        unit = normalize_unit(row.iloc[7])

        mic_value = None

        # --- Determine MIC ---
        if is_numeric(activity_value):
            mic_value = float(activity_value)

        elif is_numeric(concentration):
            mic_value = float(concentration)

        elif ">" in concentration:
            val = float(re.findall(r"\d+\.?\d*", concentration)[0])
            mic_value = val

        else:
            raise ValueError("Unknown format")

        # --- Unit conversion ---
        if unit == "µM":
            mic_value = convert_uM_to_ugml(mic_value, sequence)

        elif unit == "µg/ml":
            pass

        else:
            raise ValueError(f"Unknown unit: {unit}")

        # --- Classification ---
        if "<" in concentration:
            label = 1
        elif ">" in concentration:
            label = 0
        else:
            label = 1 if mic_value <= ACTIVE_THRESHOLD else 0

        clean_rows.append({
            "sequence": sequence,
            "species": species,
            "mic": mic_value,
            "active": label
        })

    except:
        unknown_rows.append({
            "sequence": row.iloc[3],
            "species": row.iloc[4 - 1],
            "mic": None,
            "active": None
        })


clean_df = pd.DataFrame(clean_rows)
unknown_df = pd.DataFrame(unknown_rows)

final_df = pd.concat([clean_df, unknown_df], ignore_index=True)

os.makedirs("data/processed/embeddings", exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("Done.")
print("Clean rows:", len(clean_df))
print("Unknown rows:", len(unknown_df))