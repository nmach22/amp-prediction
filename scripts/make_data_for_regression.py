import pandas as pd
import re
import os

INPUT_PATH = "data/raw/dbaasp_full.csv"
OUTPUT_PATH = "data/processed/embeddings/data_for_regression.csv"


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


# --- Parse MIC value robustly ---
def parse_mic(activity_value, concentration):
    activity_value = str(activity_value)
    concentration = str(concentration)

    # --- Case 1: clean numeric activity_value ---
    if is_numeric(activity_value):
        return float(activity_value)

    # --- Case 2: range like 3-5 ---
    if "-" in concentration:
        nums = re.findall(r"\d+\.?\d*", concentration)
        if len(nums) == 2:
            return (float(nums[0]) + float(nums[1])) / 2

    # --- Case 3: ± format like 3.1 ± 1.5 ---
    if "±" in concentration or "+-" in concentration:
        num = re.findall(r"\d+\.?\d*", concentration)
        if len(num) >= 1:
            return float(num[0])

    # --- Case 4: simple numeric concentration fallback ---
    if is_numeric(concentration):
        return float(concentration)

    # --- Case 5: censored data → drop ---
    if ">" in concentration or "<" in concentration:
        return None

    return None


# --- Main ---
df = pd.read_csv(INPUT_PATH)

# Keep only MIC rows
df = df[df.iloc[:, 4] == "MIC"].copy()

clean_rows = []
unknown_rows = []

for _, row in df.iterrows():
    try:
        sequence = row.iloc[2]
        species = row.iloc[3]
        activity_value = row.iloc[5]
        concentration = row.iloc[6]
        unit = row.iloc[7]

        mic_value = parse_mic(activity_value, concentration)

        if mic_value is None:
            raise ValueError("Unusable MIC")

        # --- Unit conversion ---
        if unit == "µM":
            mic_value = convert_uM_to_ugml(mic_value, sequence)

        elif unit == "µg/ml":
            pass

        else:
            raise ValueError(f"Unknown unit: {unit}")

        clean_rows.append({
            "sequence": sequence,
            "species": species,
            "mic": mic_value
        })

    except:
        unknown_rows.append({
            "sequence": row.iloc[2],
            "species": row.iloc[3],
            "mic": None
        })


# --- Combine ---
clean_df = pd.DataFrame(clean_rows)
unknown_df = pd.DataFrame(unknown_rows)

final_df = pd.concat([clean_df, unknown_df], ignore_index=True)

# --- Save ---
os.makedirs("data/processed/embeddings", exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("Done.")
print("Clean rows:", len(clean_df))
print("Unknown rows:", len(unknown_df))