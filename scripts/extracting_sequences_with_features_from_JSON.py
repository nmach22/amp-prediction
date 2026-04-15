import json
import pandas as pd
import re
import os

INPUT_PATH = "data/raw/dbaasp_full.json"
OUTPUT_PATH = "data/processed/embeddings/sequences_with_features_from_DBAASP.csv"


# --- Helpers ---
def compute_mw(sequence):
    return len(sequence) * 110


def convert_uM_to_ugml(value, sequence):
    return value * compute_mw(sequence) / 1000


def is_numeric(x):
    try:
        float(x)
        return True
    except:
        return False


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


def normalize_unit(unit):
    return str(unit).replace("μ", "µ").replace("�", "µ")


# --- Load JSON ---
with open(INPUT_PATH, "r") as f:
    data = json.load(f)


rows = []
unknown_rows = []

for peptide in data:
    sequence = peptide.get("sequence")

    # --- Extract physico-chemical features ---
    physchem = peptide.get("physicoChemicalProperties") or []
    feature_dict = {}

    for item in physchem:
        name = item.get("name")
        value = item.get("value")

        if name and value and is_numeric(value):
            clean_name = name.lower().replace(" ", "_")
            feature_dict[clean_name] = float(value)

    # --- Extra features ---
    feature_dict["sequence_length"] = len(sequence) if sequence else None
    feature_dict["n_term"] = peptide.get("nTerminus", {}).get("name") if peptide.get("nTerminus") else None
    feature_dict["c_term"] = peptide.get("cTerminus", {}).get("name") if peptide.get("cTerminus") else None

    # --- Iterate activities ---
    for act in peptide.get("targetActivities", []):
        try:
            measure_group = act.get("activityMeasureGroup")
            if not measure_group or measure_group.get("name") != "MIC":
                continue

            # species
            species_obj = act.get("targetSpecies")
            species = species_obj.get("name") if species_obj else None
            if species is None:
                raise ValueError("No species")

            # unit
            unit_obj = act.get("unit")
            unit = normalize_unit(unit_obj.get("name")) if unit_obj else None
            if unit is None:
                raise ValueError("No unit")

            activity_value = act.get("activity")
            concentration = act.get("concentration")
            reference = act.get("reference")

            mic = parse_mic(activity_value, concentration)
            if mic is None:
                raise ValueError("Bad MIC")

            # convert units
            if unit == "µM":
                mic = convert_uM_to_ugml(mic, sequence)
            elif unit == "µg/ml":
                pass
            else:
                raise ValueError("Unknown unit")

            row = {
                "sequence": sequence,
                "species": species,
                "mic": mic,
                "reference": reference
            }

            # add all features
            row.update(feature_dict)

            rows.append(row)

        except:
            unknown_rows.append({
                "sequence": sequence,
                "species": species if 'species' in locals() else None,
                "mic": None
            })


# --- Save ---
df = pd.DataFrame(rows)
unknown_df = pd.DataFrame(unknown_rows)

final_df = pd.concat([df, unknown_df], ignore_index=True)

os.makedirs("data/processed/embeddings", exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("Done.")
print("Rows:", len(df))
print("Unknown:", len(unknown_df))