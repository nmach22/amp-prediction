"""
run_experiment.py
=================
Single CLI entry-point for all AMP prediction experiments.

Every experiment is fully described by a YAML config file in experiments/.
Results (params, metrics, figures, model artefacts) are logged to MLflow.

Usage:
    python run_experiment.py --config experiments/rf_physicochemical.yml
    python run_experiment.py --config experiments/svm_word2vec.yml
    python run_experiment.py --config experiments/esm2_lr.yml

Then inspect results:
    mlflow ui          # opens browser at http://127.0.0.1:5000
"""

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np

# ── make src importable ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data import load_split
from src.features import (
    OneHotEncoder,
    PhysicochemicalEncoder,
    Word2VecEncoder,
    PLMEncoder,
)
from src.models import SklearnModel
from src.evaluation import compute_metrics, plot_roc_curve, plot_confusion_matrix
from src.utils import load_config, set_seed, get_logger

log = get_logger(__name__)

# ── Feature factory ──────────────────────────────────────────────────────────

def build_features(cfg: dict, train_seqs: list, val_seqs: list, test_seqs: list):
    """Instantiate encoder, fit on train, transform all splits."""
    name   = cfg["features"]["name"]
    params = cfg["features"].get("params", {}) or {}

    if name == "onehot":
        enc = OneHotEncoder(**params)
        return enc.encode(train_seqs), enc.encode(val_seqs), enc.encode(test_seqs)

    if name == "physicochemical":
        enc = PhysicochemicalEncoder()
        return enc.encode(train_seqs), enc.encode(val_seqs), enc.encode(test_seqs)

    if name == "word2vec":
        enc = Word2VecEncoder(**params)
        enc.fit(train_seqs)
        return enc.encode(train_seqs), enc.encode(val_seqs), enc.encode(test_seqs)

    if name == "plm":
        enc = PLMEncoder(**params)
        return enc.encode(train_seqs), enc.encode(val_seqs), enc.encode(test_seqs)

    raise ValueError(f"Unknown feature encoder: '{name}'")

# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an AMP prediction experiment.")
    p.add_argument(
        "--config", required=True,
        help="Path to the experiment YAML (e.g. experiments/rf_physicochemical.yml)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # ── Seed ─────────────────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # ── Load fixed splits ────────────────────────────────────────────────────
    log.info("Loading splits …")
    df_train = load_split("train")
    df_val   = load_split("val")
    df_test  = load_split("test")

    train_seqs = df_train["sequence"].tolist()
    val_seqs   = df_val["sequence"].tolist()
    test_seqs  = df_test["sequence"].tolist()

    y_train = df_train["activity"].values
    y_val   = df_val["activity"].values
    y_test  = df_test["activity"].values

    # ── Feature engineering ───────────────────────────────────────────────────
    log.info(f"Encoding features: {cfg['features']['name']} …")
    X_train, X_val, X_test = build_features(cfg, train_seqs, val_seqs, test_seqs)

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow_group = cfg.get("mlflow_experiment_group", "default")
    mlflow.set_experiment(mlflow_group)

    with mlflow.start_run(run_name=cfg["experiment_name"]):

        # Log config metadata
        mlflow.set_tags(cfg.get("tags", {}))
        mlflow.log_param("config_file", args.config)
        mlflow.log_param("seed", seed)
        mlflow.log_param("feature_encoder", cfg["features"]["name"])
        mlflow.log_param("model_name", cfg["model"]["name"])

        model_params = cfg["model"].get("params", {}) or {}
        mlflow.log_params({f"model_{k}": v for k, v in model_params.items()})

        feature_params = cfg["features"].get("params", {}) or {}
        mlflow.log_params({f"feat_{k}": v for k, v in feature_params.items()})

        # ── Train ────────────────────────────────────────────────────────────
        log.info(f"Training {cfg['model']['name']} …")
        model = SklearnModel(name=cfg["model"]["name"], params=model_params)
        model.fit(X_train, y_train)

        # ── Evaluate ─────────────────────────────────────────────────────────
        for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)
            metrics = compute_metrics(y, y_pred, y_prob)

            log.info(f"[{split_name}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            mlflow.log_metrics({f"{split_name}_{k}": v for k, v in metrics.items()})

            # Figures → logged as MLflow artefacts
            roc_fig = plot_roc_curve(y, y_prob, label=cfg["experiment_name"])
            mlflow.log_figure(roc_fig, f"{split_name}_roc_curve.png")

            cm_fig = plot_confusion_matrix(y, y_pred)
            mlflow.log_figure(cm_fig, f"{split_name}_confusion_matrix.png")

        # ── Log config file as artefact ───────────────────────────────────────
        mlflow.log_artifact(args.config, artifact_path="config")

        log.info(f"Run logged to MLflow experiment: '{mlflow_group}'")
        log.info("Run: mlflow ui  to explore results.")


if __name__ == "__main__":
    main()

