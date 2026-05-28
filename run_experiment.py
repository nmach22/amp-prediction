"""
run_experiment.py
=================
Single CLI entry-point for all AMP prediction experiments.

Every experiment is fully described by a YAML config file in experiments/.
Results (params, metrics, figures, model artifacts) are logged to W&B.

Usage:
    python run_experiment.py --config experiments/rf_physicochemical.yml
    python run_experiment.py --config experiments/svm_word2vec.yml
    python run_experiment.py --config experiments/esm2_lr.yml

"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── make src importable ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data import load_split
import src.features as feature_encoders
from src.models import SklearnModel
from src.evaluation import compute_metrics, plot_roc_curve, plot_confusion_matrix
from src.utils import (
    set_seed,
    get_logger,
    log_wandb_run,
    resolve_wandb_settings,
)

log = get_logger(__name__)

# ── Feature factory ──────────────────────────────────────────────────────────

def build_features(cfg: dict, train_seqs: list, val_seqs: list, test_seqs: list):
    """Instantiate encoder, fit on train, transform all splits."""
    name   = cfg["features"]["name"]
    params = cfg["features"].get("params", {}) or {}

    if name == "onehot":
        enc = feature_encoders.OneHotEncoder(**params)
        return enc.encode(train_seqs), enc.encode(val_seqs), enc.encode(test_seqs)

    if name == "physicochemical":
        enc = feature_encoders.PhysicochemicalEncoder()
        return enc.encode(train_seqs), enc.encode(val_seqs), enc.encode(test_seqs)

    if name == "word2vec":
        enc = feature_encoders.Word2VecEncoder(**params)
        enc.fit(train_seqs)
        return enc.encode(train_seqs), enc.encode(val_seqs), enc.encode(test_seqs)

    if name == "plm":
        enc = feature_encoders.PLMEncoder(**params)
        return enc.encode(train_seqs), enc.encode(val_seqs), enc.encode(test_seqs)

    raise ValueError(f"Unknown feature encoder: '{name}'")

# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an AMP prediction experiment.")
    p.add_argument(
        "--config", required=True,
        help="Path to the experiment YAML (e.g. experiments/rf_physicochemical.yml)"
    )
    p.add_argument(
        "--output-dir",
        default="results",
        help="Directory for model, predictions, figures, and metric outputs.",
    )
    p.add_argument(
        "--wandb-project",
        default=None,
        help="Weights & Biases project. Overrides config and config/wandb.yml.",
    )
    p.add_argument(
        "--wandb-mode",
        default=None,
        choices=["online", "offline", "disabled"],
        help="Optional W&B mode. Overrides config/wandb.yml when provided.",
    )
    p.add_argument(
        "--wandb-config",
        default="config/wandb.yml",
        help="Path to local W&B YAML config.",
    )
    p.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Skip Weights & Biases logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src.utils import load_config

    cfg  = load_config(args.config)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

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

    model_params = cfg["model"].get("params", {}) or {}
    feature_params = cfg["features"].get("params", {}) or {}

    # ── Train ────────────────────────────────────────────────────────────────
    log.info(f"Training {cfg['model']['name']} …")
    model = SklearnModel(name=cfg["model"]["name"], params=model_params)
    model.fit(X_train, y_train)

    model_path = models_dir / f"{cfg['experiment_name']}_model.joblib"
    joblib.dump(
        {
            "model": model._clf,
            "config": cfg,
            "config_file": args.config,
        },
        model_path,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    metrics_by_split: dict[str, dict[str, float]] = {}
    prediction_frames = []
    for split_name, X, y, split_df in [
        ("train", X_train, y_train, df_train),
        ("val", X_val, y_val, df_val),
        ("test", X_test, y_test, df_test),
    ]:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        metrics = compute_metrics(y, y_pred, y_prob)
        metrics_by_split[split_name] = metrics

        log.info(
            f"[{split_name}] "
            + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

        pred_df = split_df[["sequence", "activity"]].copy()
        pred_df["split"] = split_name
        pred_df["pred"] = y_pred
        pred_df["prob_amp"] = y_prob
        prediction_frames.append(pred_df)

        roc_fig = plot_roc_curve(y, y_prob, label=cfg["experiment_name"])
        roc_fig.savefig(figures_dir / f"{cfg['experiment_name']}_{split_name}_roc_curve.png")
        plt.close(roc_fig)

        cm_fig = plot_confusion_matrix(y, y_pred)
        cm_fig.savefig(figures_dir / f"{cfg['experiment_name']}_{split_name}_confusion_matrix.png")
        plt.close(cm_fig)

    pd.DataFrame(metrics_by_split).T.to_csv(
        tables_dir / f"{cfg['experiment_name']}_metrics.csv",
        index_label="split",
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        tables_dir / f"{cfg['experiment_name']}_predictions.csv",
        index=False,
    )

    run_config = {
        "config_file": args.config,
        "output_dir": str(output_dir),
        "seed": seed,
        "feature_encoder": cfg["features"]["name"],
        "model_name": cfg["model"]["name"],
        **{f"model_{k}": v for k, v in model_params.items()},
        **{f"feat_{k}": v for k, v in feature_params.items()},
    }
    wandb_settings = resolve_wandb_settings(
        config_path=args.wandb_config,
        default_project=cfg.get("wandb_project", "amp-prediction"),
        cli_project=args.wandb_project,
        cli_mode=args.wandb_mode,
        cli_disabled=args.disable_wandb,
    )
    if wandb_settings["enabled"]:
        log_wandb_run(
            project=wandb_settings["project"],
            run_name=cfg["experiment_name"],
            config=run_config,
            metrics_by_split=metrics_by_split,
            mode=wandb_settings["mode"],
            entity=wandb_settings["entity"],
            tags=list(cfg.get("tags", {}).values()) + wandb_settings["tags"],
            api_key=wandb_settings["api_key"],
            figures_dir=figures_dir,
        )
        log.info("Logged run to W&B project: %s", wandb_settings["project"])


if __name__ == "__main__":
    main()
