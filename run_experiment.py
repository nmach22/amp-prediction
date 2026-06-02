"""
run_experiment.py
=================
Single CLI entry-point for all AMP prediction experiments.

Classification experiments are described by YAML config files in experiments/.
MIC baselines are selected by model name. Results, metrics, and artifacts are
logged to W&B when enabled.

Usage:
    python run_experiment.py --config experiments/rf_physicochemical.yml
    python run_experiment.py --config experiments/svm_word2vec.yml
    python run_experiment.py --config experiments/esm2_lr.yml
    python run_experiment.py --model mic_baseline --input data/processed/splits/train.csv
    python run_experiment.py --model taxonomy_mic_baseline --input data/processed/splits/train.csv

"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── make src importable ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data import load_split
import src.features as feature_encoders
from src.models.registry import MIC_EXPERIMENT_NAMES
from src.utils import (
    set_seed,
    get_logger,
    log_wandb_run,
    resolve_wandb_settings,
)

log = get_logger(__name__)
DEFAULT_ESTIMATOR_CHECKPOINTS = (1, 5, 10, 25, 50, 100, 200)

# ── Feature factory ──────────────────────────────────────────────────────────

def build_features(cfg: dict, train_seqs: list, val_seqs: list):
    """Instantiate encoder, fit on train when needed, and transform train/val."""
    name   = cfg["features"]["name"]
    params = cfg["features"].get("params", {}) or {}

    if name == "onehot":
        enc = feature_encoders.OneHotEncoder(**params)
        return enc, enc.encode(train_seqs), enc.encode(val_seqs)

    if name == "physicochemical":
        enc = feature_encoders.PhysicochemicalEncoder()
        return enc, enc.encode(train_seqs), enc.encode(val_seqs)

    if name == "word2vec":
        enc = feature_encoders.Word2VecEncoder(**params)
        enc.fit(train_seqs)
        return enc, enc.encode(train_seqs), enc.encode(val_seqs)

    if name == "plm":
        enc = feature_encoders.PLMEncoder(**params)
        return enc, enc.encode(train_seqs), enc.encode(val_seqs)

    raise ValueError(f"Unknown feature encoder: '{name}'")


def make_serializable_encoder(encoder):
    """Drop lazy-loaded PLM objects before saving the inference bundle."""
    if hasattr(encoder, "_model"):
        encoder._model = None
    if hasattr(encoder, "_tokenizer"):
        encoder._tokenizer = None
    return encoder


def estimator_checkpoints(n_estimators: int) -> list[int]:
    checkpoints = {
        checkpoint
        for checkpoint in DEFAULT_ESTIMATOR_CHECKPOINTS
        if checkpoint <= n_estimators
    }
    checkpoints.add(n_estimators)
    return sorted(checkpoints)


def collect_classification_history(
    model,
    split_data: list[tuple[str, np.ndarray, np.ndarray]],
    step: int,
    num_estimators: int | None = None,
) -> list[dict]:
    from src.evaluation.metrics import compute_metrics

    rows = []
    for split_name, X, y in split_data:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)
        row = {
            "step": step,
            "split": split_name,
            "metrics": compute_metrics(y, y_pred, y_prob),
        }
        if num_estimators is not None:
            row["num_estimators"] = num_estimators
        rows.append(row)
    return rows


def build_wandb_run_config(
    *,
    cfg: dict,
    config_file: str,
    output_dir: Path,
    seed: int,
    model=None,
) -> dict:
    """Build W&B config with both requested and resolved model settings."""
    model_config = cfg["model"]
    feature_config = cfg["features"]
    model_params = model_config.get("params", {}) or {}
    feature_params = feature_config.get("params", {}) or {}

    resolved_model_params = {}
    if model is not None and hasattr(model._clf, "get_params"):
        resolved_model_params = model._clf.get_params(deep=False)

    return {
        "experiment_name": cfg.get("experiment_name"),
        "config_file": config_file,
        "output_dir": str(output_dir),
        "seed": seed,
        "tags": cfg.get("tags", {}),
        "feature_encoder": feature_config["name"],
        "feature_config": feature_config,
        "model_name": model_config["name"],
        "model_config": model_config,
        "resolved_model_params": resolved_model_params,
        **{f"model_{k}": v for k, v in model_params.items()},
        **{f"feat_{k}": v for k, v in feature_params.items()},
        **{
            f"resolved_model_{k}": v
            for k, v in resolved_model_params.items()
        },
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an AMP prediction experiment.")
    mic_models = sorted(MIC_EXPERIMENT_NAMES)
    p.add_argument(
        "--config",
        default=None,
        help="Path to the experiment YAML (e.g. experiments/rf_physicochemical.yml)"
    )
    p.add_argument(
        "--model",
        default=None,
        choices=mic_models,
        help=(
            "Named MIC regression baseline to run. Available: "
            f"{', '.join(mic_models)}"
        ),
    )
    p.add_argument(
        "--input",
        default="data/processed/splits/train.csv",
        help="Input CSV for named MIC regression baselines.",
    )
    p.add_argument("--seed", type=int, default=None, help="Override random seed.")
    p.add_argument(
        "--run-name",
        default=None,
        help="Weights & Biases run name for named MIC baselines.",
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
    args = p.parse_args()
    if bool(args.config) == bool(args.model):
        p.error("Provide exactly one of --config or --model.")
    return args


def main() -> None:
    args = parse_args()
    if args.model:
        run_named_mic_experiment(args)
    else:
        run_classification_experiment(args)


def run_classification_experiment(args: argparse.Namespace) -> None:
    from src.utils import load_config
    from src.models import SklearnModel
    from src.evaluation.metrics import compute_metrics
    import matplotlib.pyplot as plt
    from src.evaluation import plot_confusion_matrix, plot_roc_curve

    cfg  = load_config(args.config)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Seed ─────────────────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    # ── Load fixed splits ────────────────────────────────────────────────────
    log.info("Loading splits …")
    df_train = load_split("train")
    df_val   = load_split("val")

    train_seqs = df_train["sequence"].tolist()
    val_seqs   = df_val["sequence"].tolist()

    y_train = df_train["activity"].values
    y_val   = df_val["activity"].values

    # ── Feature engineering ───────────────────────────────────────────────────
    log.info(f"Encoding features: {cfg['features']['name']} …")
    encoder, X_train, X_val = build_features(cfg, train_seqs, val_seqs)

    model_params = cfg["model"].get("params", {}) or {}

    # ── Train ────────────────────────────────────────────────────────────────
    log.info(f"Training {cfg['model']['name']} …")
    model = SklearnModel(name=cfg["model"]["name"], params=model_params)
    split_data = [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
    ]
    metric_history: list[dict] = []
    if cfg["model"]["name"] == "random_forest":
        total_estimators = model._clf.n_estimators
        model._clf.set_params(warm_start=True)
        for step, n_estimators in enumerate(
            estimator_checkpoints(total_estimators), start=1
        ):
            model._clf.set_params(n_estimators=n_estimators)
            model.fit(X_train, y_train)
            metric_history.extend(
                collect_classification_history(
                    model,
                    split_data,
                    step=step,
                    num_estimators=n_estimators,
                )
            )
    else:
        model.fit(X_train, y_train)
        metric_history.extend(
            collect_classification_history(model, split_data, step=1)
        )

    model_path = models_dir / f"{cfg['experiment_name']}_model.joblib"
    joblib.dump(
        {
            "model": model._clf,
            "feature_encoder": make_serializable_encoder(encoder),
            "config": cfg,
            "config_file": args.config,
            "trained_splits": ["train", "val"],
        },
        model_path,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    metrics_by_split: dict[str, dict[str, float]] = {}
    prediction_frames = []
    for split_name, X, y, split_df in [
        ("train", X_train, y_train, df_train),
        ("val", X_val, y_val, df_val),
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

    run_config = build_wandb_run_config(
        cfg=cfg,
        config_file=args.config,
        output_dir=output_dir,
        seed=seed,
        model=model,
    )
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
            metric_history=metric_history,
            mode=wandb_settings["mode"],
            entity=wandb_settings["entity"],
            tags=list(cfg.get("tags", {}).values()) + wandb_settings["tags"],
            api_key=wandb_settings["api_key"],
            figures_dir=figures_dir,
        )
        log.info("Logged run to W&B project: %s", wandb_settings["project"])


def run_named_mic_experiment(args: argparse.Namespace) -> None:
    from src.models.mic_runner import train_and_evaluate_mic_baseline
    from src.models.registry import get_mic_experiment_spec

    spec = get_mic_experiment_spec(args.model)
    output_dir = Path(args.output_dir)
    seed = args.seed if args.seed is not None else 42
    set_seed(seed)

    metrics, metric_history = train_and_evaluate_mic_baseline(
        spec=spec,
        input_csv=Path(args.input),
        output_dir=output_dir,
        random_state=seed,
        return_history=True,
    )

    wandb_settings = resolve_wandb_settings(
        config_path=args.wandb_config,
        default_project=spec.default_project,
        cli_project=args.wandb_project,
        cli_mode=args.wandb_mode,
        cli_disabled=args.disable_wandb,
    )

    run_config = {
        "input_csv": args.input,
        "output_dir": str(output_dir),
        "seed": seed,
        "model": args.model,
        **spec.run_config,
    }
    if wandb_settings["enabled"]:
        log_wandb_run(
            project=wandb_settings["project"],
            run_name=args.run_name or spec.default_run_name,
            config=run_config,
            metrics_by_split=metrics,
            metric_history=metric_history,
            mode=wandb_settings["mode"],
            entity=wandb_settings["entity"],
            tags=wandb_settings["tags"],
            api_key=wandb_settings["api_key"],
        )

    for split, split_metrics in metrics.items():
        metric_names = ["mae", "rmse", "r2"]
        formatted = " | ".join(
            f"{name}={split_metrics[name]:.4f}"
            for name in metric_names
            if name in split_metrics
        )
        log.info("%s | %s", split, formatted)
    log.info("Saved %s outputs to %s", args.model, output_dir.resolve())


if __name__ == "__main__":
    main()
