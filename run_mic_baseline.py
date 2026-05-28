"""
Train a simple baseline model for MIC regression.

Usage:
    python run_mic_baseline.py
    python run_mic_baseline.py --input data/processed/splits/train.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.models.mic_baseline import train_and_evaluate
from src.utils import log_wandb_run, resolve_wandb_settings

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
        default="data/processed/splits/train.csv",
        help="Training CSV with sequence, gram_status, and activity columns.",
    )
    parser.add_argument(
        "--test-input",
        default=None,
        help=(
            "Optional held-out test CSV. If omitted and --input is train.csv, "
            "a sibling test.csv is used when present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for model, predictions, and metric outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--run-name",
        default="mic_baseline_random_forest",
        help="W&B run name.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help=(
            "Weights & Biases project. Overrides config/wandb.yml when provided."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode. Overrides config/wandb.yml when provided.",
    )
    parser.add_argument(
        "--wandb-config",
        default="config/wandb.yml",
        help="Path to local W&B YAML config.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Skip Weights & Biases logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    metrics = train_and_evaluate(
        input_csv=Path(args.input),
        output_dir=output_dir,
        random_state=args.seed,
        test_csv=Path(args.test_input) if args.test_input else None,
    )
    wandb_settings = resolve_wandb_settings(
        config_path=args.wandb_config,
        default_project="mic-baseline",
        cli_project=args.wandb_project,
        cli_mode=args.wandb_mode,
        cli_disabled=args.disable_wandb,
    )

    run_config = {
        "input_csv": args.input,
        "test_input_csv": args.test_input or "auto",
        "output_dir": str(output_dir),
        "seed": args.seed,
        "model_name": "random_forest_regressor",
        "target": "log10_mic",
    }

    if wandb_settings["enabled"]:
        log_wandb_run(
            project=wandb_settings["project"],
            run_name=args.run_name,
            config=run_config,
            metrics_by_split=metrics,
            mode=wandb_settings["mode"],
            entity=wandb_settings["entity"],
            tags=wandb_settings["tags"],
            api_key=wandb_settings["api_key"],
        )

    for split, split_metrics in metrics.items():
        formatted = " | ".join(
            f"{name}={value:.4f}" for name, value in split_metrics.items()
        )
        log.info("%s | %s", split, formatted)
    log.info("Saved outputs to %s", output_dir.resolve())
    if wandb_settings["enabled"]:
        log.info("Logged run to W&B project: %s", wandb_settings["project"])


if __name__ == "__main__":
    main()
