"""
Train a simple baseline model for MIC regression.

Usage:
    python run_mic_baseline.py
    python run_mic_baseline.py --input data/processed/amp_mic_activities_gram_classified.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.mic_baseline import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MIC regression baseline.")
    parser.add_argument(
        "--input",
        default="data/processed/amp_mic_activities_gram_classified.csv",
        help="CSV with sequence, gram_status, and activity columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for model, predictions, and metric outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(
        input_csv=Path(args.input),
        output_dir=Path(args.output_dir),
        random_state=args.seed,
    )
    for split, split_metrics in metrics.items():
        formatted = " | ".join(
            f"{name}={value:.4f}" for name, value in split_metrics.items()
        )
        log.info("%s | %s", split, formatted)
    log.info("Saved outputs to %s", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
