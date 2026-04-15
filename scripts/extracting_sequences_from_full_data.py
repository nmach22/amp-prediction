import json
import pandas as pd
import re
import os

INPUT_PATH = "data/raw/dbaasp_full.json"
OUTPUT_PATH = "data/processed/embeddings/sequences_with_MIC.csv"


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


# --- Parse MIC ---
def parse_mic(activity, concentration):
    activity = str(activity)
    concentration = str(concentration)

    if is_numeric(activity):
        return float(activity)

    if "-" in concentration:
        nums = re.findall(r"\d+\.?\d*", concentration)
        if len(nums) == 2:
            return (float(nums[0]) + float(nums[1])) / 2

    if "±" in concentration or "+-" in concentration:
        nums = re.findall(r"\d+\.?\d*", concentration)
        if len(nums) >= 1:
            return float(nums[0])

    if is_numeric(concentration):
        return float(concentration)

    if ">" in concentration or "<" in concentration:
        return None

    return None


# --- Load JSON ---
with open(INPUT_PATH, "r") as f:
    data = json.load(f)


clean_rows = []
unknown_rows = []

for peptide in data:
    sequence = peptide.get("sequence")

    for act in peptide.get("targetActivities", []):
        try:
            # --- SAFE measure check ---
            measure_group = act.get("activityMeasureGroup")
            if not measure_group or measure_group.get("name") != "MIC":
                continue

            # --- SAFE species ---
            species_obj = act.get("targetSpecies")
            species = species_obj.get("name") if species_obj else None
            if species is None:
                raise ValueError("Missing species")

            # --- SAFE unit ---
            unit_obj = act.get("unit")
            unit = unit_obj.get("name") if unit_obj else None
            if unit is None:
                raise ValueError("Missing unit")

            unit = unit.replace("μ", "µ")

            activity_value = act.get("activity")
            concentration = act.get("concentration")
            reference = act.get("reference")

            mic_value = parse_mic(activity_value, concentration)

            if mic_value is None:
                raise ValueError("Bad MIC")

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
                "mic": mic_value,
                "reference": reference
            })

        except:
            species_obj = act.get("targetSpecies")
            species = species_obj.get("name") if species_obj else None

            unknown_rows.append({
                "sequence": sequence,
                "species": species,
                "mic": None,
                "reference": act.get("reference")
            })


# --- Save ---
clean_df = pd.DataFrame(clean_rows)
unknown_df = pd.DataFrame(unknown_rows)

final_df = pd.concat([clean_df, unknown_df], ignore_index=True)

os.makedirs("data/processed/embeddings", exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("Done.")
print("Clean rows:", len(clean_df))
print("Unknown rows:", len(unknown_df))