"""Run MIC inference from a saved inference bundle and an input CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.mic_inference import predict_mic_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict MIC values using a saved MIC inference bundle.",
    )
    parser.add_argument(
        "--model",
        default="results/inference/best_mic_model.joblib",
        help="Path to MIC inference bundle.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV with at least a sequence column.",
    )
    parser.add_argument(
        "--output",
        default="results/inference/mic_predictions.csv",
        help="Output CSV for predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_mic_csv(args.model, args.input, args.output)
    print(f"Saved {len(predictions)} MIC predictions to {args.output}")


if __name__ == "__main__":
    main()
