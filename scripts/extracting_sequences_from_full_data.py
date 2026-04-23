import json
import pandas as pd
import re
import os

INPUT_PATH = "data/raw/dbaasp_full.json"
OUTPUT_PATH = "data/processed/embeddings/sequences_with_MIC.csv"


# --- Amino acid residue molecular weights (Da) ---
# These are residue weights (i.e., after losing H2O during peptide bond formation)
AA_RESIDUE_MW = {
    'A': 71.03711,
    'R': 156.10111,
    'N': 114.04293,
    'D': 115.02694,
    'C': 103.00919,
    'E': 129.04259,
    'Q': 128.05858,
    'G': 57.02146,
    'H': 137.05891,
    'I': 113.08406,
    'L': 113.08406,
    'K': 128.09496,
    'M': 131.04049,
    'F': 147.06841,
    'P': 97.05276,
    'S': 87.03203,
    'T': 101.04768,
    'W': 186.07931,
    'Y': 163.06333,
    'V': 99.06841,
    'U': 150.95364,
    'O': 237.14773,
    'B': 114.535,
    'Z': 128.551,
}

WATER_MW = 18.01056  # Add once per peptide (N- and C-terminus)

def compute_mw(sequence):
    """
    Compute peptide molecular weight from sequence.
    Sum of residue weights + water (for the free termini).
    Unknown AAs fall back to average 110 Da.
    """
    mw = WATER_MW
    for aa in sequence.upper():
        mw += AA_RESIDUE_MW.get(aa, 110.0)  # fallback for non-standard AAs
    return mw

def convert_ugml_to_uM(value, sequence):
    mw = compute_mw(sequence)
    return (value * 1000) / mw


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

            # --- Unit conversion block ---
            if unit == "µg/ml":
                mic_value = convert_ugml_to_uM(mic_value, sequence)  # convert to µM
            elif unit == "µM":
                pass  # already in µM, keep as-is
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