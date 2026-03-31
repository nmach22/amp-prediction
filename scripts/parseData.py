import pandas as pd
import re

# Input / Output files
input_file = "Antimicrobial_amps.xlsx"
output_file = "parsed_dramp.xlsx"

# Read Excel
df = pd.read_excel(input_file)

# Columns (adjust if needed)
sequence_col = df.columns[1]   # Column B
target_col = df.columns[14]    # Column O

rows = []

# Regex to extract: organism + MIC
pattern = re.compile(r'([A-Za-z0-9\.\-\s]+?)\s*\(MIC=([\d\.]+)\s*µg/ml\)')

for _, row in df.iterrows():
    sequence = row[sequence_col]
    target_text = row[target_col]

    # Skip missing data
    if pd.isna(sequence) or pd.isna(target_text):
        continue

    # Split by ## (groups like Gram+/Gram-/Yeast)
    groups = str(target_text).split("##")

    for group in groups:
        matches = pattern.findall(group)

        for organism, mic in matches:
            rows.append({
                "sequence": sequence,
                "organism": organism.strip(),
                "MIC": float(mic)
            })

# Create new dataframe
new_df = pd.DataFrame(rows)

# Save to Excel
new_df.to_excel(output_file, index=False)

print(f"Done! Saved to {output_file}")