"""Shared training runner for MIC regression baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.models.mic_baseline import build_model as build_mic_baseline_model
from src.models.mic_baseline import split_train_val_by_sequence

DEFAULT_ESTIMATOR_CHECKPOINTS = (1, 5, 10, 25, 50, 100, 200)


@dataclass(frozen=True)
class MicExperimentSpec:
    """Configuration for one MIC baseline variant."""

    name: str
    default_project: str
    default_run_name: str
    load_data: Callable[[str | Path], pd.DataFrame]
    build_features: Callable[[pd.DataFrame], pd.DataFrame]
    evaluate_predictions: Callable[
        [pd.DataFrame, np.ndarray, np.ndarray],
        dict[str, float],
    ]
    prediction_columns: tuple[str, ...]
    build_model: Callable[[int], BaseModel] = build_mic_baseline_model
    use_estimator_checkpoints: bool = True
    use_validation_fit: bool = False
    artifact_metadata: Callable[[pd.DataFrame], dict] = field(default=lambda df: {})
    run_config: dict = field(default_factory=dict)


def estimator_checkpoints(n_estimators: int) -> list[int]:
    """Return increasing Random Forest checkpoints up to n_estimators."""
    checkpoints = {
        checkpoint
        for checkpoint in DEFAULT_ESTIMATOR_CHECKPOINTS
        if checkpoint <= n_estimators
    }
    checkpoints.add(n_estimators)
    return sorted(checkpoints)


def train_and_evaluate_mic_baseline(
    spec: MicExperimentSpec,
    input_csv: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    return_history: bool = False,
) -> dict[str, dict[str, float]] | tuple[dict[str, dict[str, float]], list[dict]]:
    """Train a MIC baseline and write predictions, metrics, and model artifacts."""
    output_path = Path(output_dir)
    tables_dir = output_path / "tables"
    models_dir = output_path / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = spec.load_data(input_csv)
    splits = split_train_val_by_sequence(df, random_state=random_state)
    split_frames = [
        ("train", splits.train),
        ("val", splits.val),
    ]

    X_train = spec.build_features(splits.train)
    y_train = splits.train["log_mic"].to_numpy()
    model = spec.build_model(random_state=random_state)
    _assert_base_model(model, spec.name)

    split_features: dict[str, pd.DataFrame] = {"train": X_train}
    split_targets: dict[str, np.ndarray] = {"train": y_train}
    for split_name, split_df in split_frames:
        if split_df.empty or split_name == "train":
            continue
        split_features[split_name] = spec.build_features(split_df).reindex(
            columns=X_train.columns,
            fill_value=0.0,
        )
        split_targets[split_name] = split_df["log_mic"].to_numpy()

    metric_history: list[dict] = []
    metrics_by_split: dict[str, dict[str, float]] = {}
    if spec.use_estimator_checkpoints:
        total_estimators = model._model.n_estimators
        model._model.set_params(warm_start=True)
        training_steps = [
            (step, n_estimators)
            for step, n_estimators in enumerate(
                estimator_checkpoints(total_estimators), start=1
            )
        ]
    else:
        training_steps = [(1, None)]

    for step, n_estimators in training_steps:
        if n_estimators is not None:
            model._model.set_params(n_estimators=n_estimators)
        if spec.use_validation_fit and "val" in split_features:
            model.fit(
                X_train,
                y_train,
                X_val=split_features["val"],
                y_val=split_targets["val"],
            )
        else:
            model.fit(X_train, y_train)
        metrics_by_split = {}
        for split_name, split_df in split_frames:
            if split_df.empty:
                continue
            y_pred = model.predict(split_features[split_name])
            metrics = spec.evaluate_predictions(
                split_df,
                split_targets[split_name],
                y_pred,
            )
            metrics_by_split[split_name] = metrics
            metric_history.append(
                {
                    "step": step,
                    "split": split_name,
                    "metrics": metrics,
                }
            )
            if n_estimators is not None:
                metric_history[-1]["num_estimators"] = n_estimators

    prediction_frames = []
    prediction_columns = [
        column for column in spec.prediction_columns if column in df.columns
    ]
    for split_name, split_df in split_frames:
        if split_df.empty or split_name != "val":
            continue
        y_pred = model.predict(split_features[split_name])
        pred_df = split_df[prediction_columns].copy()
        pred_df["split"] = split_name
        pred_df["pred_log_mic"] = y_pred
        pred_df["pred_mic"] = np.power(10.0, y_pred)
        prediction_frames.append(pred_df)

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        tables_dir / f"{spec.name}_predictions.csv",
        index=False,
    )
    pd.DataFrame(metrics_by_split).T.to_csv(
        tables_dir / f"{spec.name}_metrics.csv",
        index_label="split",
    )
    joblib.dump(
        {
            "model": model,
            "feature_columns": X_train.columns.tolist(),
            "model_name": spec.name,
            **spec.artifact_metadata(df),
            **_model_artifact_metadata(model, X_train.columns.tolist()),
        },
        models_dir / f"{spec.name}_model.joblib",
    )

    if return_history:
        return metrics_by_split, metric_history
    return metrics_by_split


def _assert_base_model(model: BaseModel, model_name: str) -> None:
    if not isinstance(model, BaseModel):
        raise TypeError(f"{model_name} did not build a BaseModel instance.")


def _model_artifact_metadata(model: BaseModel, feature_columns: list[str]) -> dict:
    metadata_fn = getattr(model, "artifact_metadata", None)
    if not callable(metadata_fn):
        return {}
    return metadata_fn(feature_columns)
