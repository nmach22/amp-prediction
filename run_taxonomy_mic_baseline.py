"""Train MIC baseline using target-species taxonomy features.

Usage:
    python run_taxonomy_mic_baseline.py
    python run_taxonomy_mic_baseline.py --input data/processed/splits/train.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.models.taxonomy_mic_baseline import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the taxonomy-feature MIC regression baseline."
    )
    parser.add_argument(
        "--input",
        default="data/processed/splits/train.csv",
        help="Training CSV with sequence, target_activity_name, activity, and taxonomy features.",
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
        "--mlflow-experiment",
        default="taxonomy-mic-baseline",
        help="MLflow experiment name for logged metrics and artifacts.",
    )
    parser.add_argument(
        "--run-name",
        default="taxonomy_mic_baseline_random_forest",
        help="MLflow run name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    try:
        import joblib
        import mlflow
        import mlflow.sklearn
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "MLflow logging requires the project environment. "
            "Run `conda activate amp` and try again."
        ) from exc

    metrics = train_and_evaluate(
        input_csv=Path(args.input),
        output_dir=output_dir,
        random_state=args.seed,
        test_csv=Path(args.test_input) if args.test_input else None,
    )

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "input_csv": args.input,
                "test_input_csv": args.test_input or "auto",
                "output_dir": str(output_dir),
                "seed": args.seed,
                "model_name": "random_forest_regressor",
                "target": "log10_mic",
                "target_features": "taxonomy",
            }
        )
        for split, split_metrics in metrics.items():
            mlflow.log_metrics(
                {f"{split}_{name}": value for name, value in split_metrics.items()}
            )

        tables_dir = output_dir / "tables"
        model_path = output_dir / "taxonomy_mic_baseline_model.joblib"
        if tables_dir.exists():
            mlflow.log_artifacts(str(tables_dir), artifact_path="tables")
        if model_path.exists():
            mlflow.log_artifact(str(model_path), artifact_path="model")
            model_payload = joblib.load(model_path)
            input_example = pd.DataFrame(
                [[0.0] * len(model_payload["feature_columns"])],
                columns=model_payload["feature_columns"],
            )
            mlflow.sklearn.log_model(
                sk_model=model_payload["model"]._model,
                name="sklearn_model",
                input_example=input_example,
            )

    for split, split_metrics in metrics.items():
        formatted = " | ".join(
            f"{name}={value:.4f}"
            for name, value in split_metrics.items()
            if name in {"mae", "rmse", "r2"}
        )
        log.info("%s | %s", split, formatted)
    log.info("Saved taxonomy baseline outputs to %s", output_dir.resolve())
    log.info("Logged run to MLflow experiment: %s", args.mlflow_experiment)


if __name__ == "__main__":
    main()
