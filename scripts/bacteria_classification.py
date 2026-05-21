"""Extract taxonomy features for MIC target organisms.

Usage:
    python scripts/bacteria_classification.py
    python scripts/bacteria_classification.py --input data/raw/amp_mic_activities.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.taxonomy import write_taxonomy_feature_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build bacterial taxonomy features from target_activity_name."
    )
    parser.add_argument(
        "--input",
        default="data/raw/amp_mic_activities.csv",
        help="CSV containing target_activity_name.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/amp_mic_activities_taxonomy_features.csv",
        help="Output CSV with taxonomy columns and one-hot taxonomy features.",
    )
    parser.add_argument(
        "--target-col",
        default="target_activity_name",
        help="Column containing target species or organism names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_taxonomy_feature_file(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        target_col=args.target_col,
    )
    feature_count = len(
        [
            column
            for column in output.columns
            if column.startswith(("Phylum_", "Class_", "Order_", "Family_", "Genus_"))
            or column == "target_is_bacteria"
        ]
    )
    print(f"Saved {len(output)} rows with {feature_count} taxonomy features to {args.output}")


if __name__ == "__main__":
    main()
