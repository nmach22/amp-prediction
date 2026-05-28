"""Evaluate a saved classifier bundle on the held-out test split.

Use this only after model selection is complete.

Usage:
    python evaluate_saved_model.py --model results/models/rf_physicochemical_model.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data import load_split
from src.evaluation import compute_metrics, plot_confusion_matrix, plot_roc_curve
from src.utils import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one selected saved model on the held-out test split."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a joblib model bundle saved by run_experiment.py.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test"],
        help="Held-out split to evaluate. Only test is supported.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/final_test",
        help="Directory for final test metrics, predictions, and figures.",
    )
    return parser.parse_args()


def positive_class_probability(model, X_test):
    probabilities = model.predict_proba(X_test)
    if hasattr(model, "classes_") and 1 in model.classes_:
        positive_index = list(model.classes_).index(1)
    else:
        positive_index = 1
    return probabilities[:, positive_index]


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)
    model = bundle["model"]
    encoder = bundle.get("feature_encoder")
    if encoder is None:
        raise ValueError(
            "Saved model bundle does not include feature_encoder. "
            "Re-run the experiment with the updated run_experiment.py."
        )

    df_test = load_split(args.split)
    test_seqs = df_test["sequence"].tolist()
    y_test = df_test["activity"].values
    X_test = encoder.encode(test_seqs)

    y_pred = model.predict(X_test)
    y_prob = positive_class_probability(model, X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)

    run_name = model_path.stem
    pd.DataFrame({"test": metrics}).T.to_csv(
        tables_dir / f"{run_name}_test_metrics.csv",
        index_label="split",
    )

    pred_df = df_test[["sequence", "activity"]].copy()
    pred_df["split"] = args.split
    pred_df["pred"] = y_pred
    pred_df["prob_amp"] = y_prob
    pred_df.to_csv(tables_dir / f"{run_name}_test_predictions.csv", index=False)

    roc_fig = plot_roc_curve(y_test, y_prob, label=run_name)
    roc_fig.savefig(figures_dir / f"{run_name}_test_roc_curve.png")
    plt.close(roc_fig)

    cm_fig = plot_confusion_matrix(y_test, y_pred)
    cm_fig.savefig(figures_dir / f"{run_name}_test_confusion_matrix.png")
    plt.close(cm_fig)

    log.info(
        "[test] " + " | ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    )
    log.info("Saved final test outputs to %s", output_dir.resolve())


if __name__ == "__main__":
    main()
