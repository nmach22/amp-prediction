# Data Feature Engineering

This document describes the feature engineering steps applied to the raw DBAASP
antimicrobial peptide (AMP) activity data before model training.

## Source data

| File | Description |
|---|---|
| `raw/amp_mic_activities.csv` | Raw MIC activity records exported from DBAASP. Columns: `sequence`, `target_activity_name`, `activity`. Contains **114 880** rows. |

## Pipeline overview

```
raw/amp_mic_activities.csv
  │
  ├─► scripts/extract_bacteria_genus.py   →  interim/amp_mic_with_genus.csv
  │       Extracts first two words of target_activity_name
  │       as a new "bacteria_genus" column (Genus + species).
  │
  └─► scripts/filter_by_genus.py          →  processed/amp_mic_filtered.csv
          1. Extracts genus label (first word, title-cased).
          2. Removes non-bacterial (fungi) organisms.
          3. Keeps only genera with ≥ N data points (default 5 000).
```

## Step 1 — Genus extraction (`extract_bacteria_genus.py`)

The `target_activity_name` column contains strain-level organism names such as
*Staphylococcus aureus ATCC 6538P*. The first two whitespace-delimited words are
extracted as the **bacteria_genus** column (e.g. *Staphylococcus aureus*), giving
a clean genus + species pair without strain or collection identifiers.

- **Input:** `raw/amp_mic_activities.csv` (114 880 rows)
- **Output:** `interim/amp_mic_with_genus.csv` (114 880 rows, +1 column)
- **Unique genus+species pairs:** 823

```bash
python scripts/extract_bacteria_genus.py
```

## Step 2 — Fungi exclusion and genus filtering (`filter_by_genus.py`)

### 2a. Fungi removal

The dataset contains eukaryotic organisms that are not bacteria. The following
**9 fungal genera** were identified and excluded:

| Genus | Type | Rows removed |
|---|---|---|
| Candida | Yeast | 6 797 |
| Aspergillus | Mold | 772 |
| Cryptococcus | Yeast | 754 |
| Fusarium | Mold | 612 |
| Saccharomyces | Yeast | 378 |
| Penicillium | Mold | 178 |
| Botrytis | Mold | 173 |
| Trichophyton | Dermatophyte | 172 |
| Trichosporon | Yeast-like fungus | 122 |
| **Total** | | **9 958** |

### 2b. Minimum-count genus filter

After fungi removal, genera with fewer than 5 000 data points are grouped out to
ensure sufficient representation for modelling. The remaining genera are stored in
the **genus_label** column.

| Genus | Rows |
|---|---|
| Staphylococcus | 25 709 |
| Escherichia | 20 915 |
| Pseudomonas | 14 424 |
| Bacillus | 6 356 |
| Klebsiella | 5 595 |
| **Total** | **72 999** |

The threshold can be adjusted with `--min-count`:

```bash
# Default (≥ 5 000 samples per genus)
python scripts/filter_by_genus.py

# Lower threshold to include more genera
python scripts/filter_by_genus.py --min-count 1000
```

- **Input:** `raw/amp_mic_activities.csv` (114 880 rows)
- **Output:** `processed/amp_mic_filtered.csv` (72 999 rows, +1 column)

## Output files

| File | Rows | Columns | Description |
|---|---|---|---|
| `interim/amp_mic_with_genus.csv` | 114 880 | sequence, target_activity_name, activity, bacteria_genus | Raw + genus+species column |
| `processed/amp_mic_filtered.csv` | 72 999 | sequence, target_activity_name, activity, genus_label | Bacteria-only, top-5 genera |

## Reproducibility

Run both steps from the project root:

```bash
python scripts/extract_bacteria_genus.py
python scripts/filter_by_genus.py
```

No external dependencies beyond pandas are required.
