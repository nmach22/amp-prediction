"""
Baseline MIC regression pipeline for AMP sequences.

The pipeline predicts log10(MIC) from simple sequence composition features
and gram status. Splits are grouped by sequence to avoid duplicate-sequence
leakage across train and validation sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from src.models import BaseModel

STANDARD_AA = tuple("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA_SET = set(STANDARD_AA)
NONSTANDARD_PATTERN = re.compile(f"[^{''.join(STANDARD_AA)}]")
GRAM_CLASSES = {"gram_positive", "gram_negative"}
RESIDUE_GROUPS = {
    "frac_positive": set("KRH"),
    "frac_negative": set("DE"),
    "frac_polar": set("STNQCY"),
    "frac_hydrophobic": set("AILMFWV"),
    "frac_aromatic": set("FWY"),
}
DEFAULT_ESTIMATOR_CHECKPOINTS = (1, 5, 10, 25, 50, 100, 200)


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_mic_data(path: str | Path) -> pd.DataFrame:
    """Load and clean MIC regression rows."""
    df = pd.read_csv(path)
    required = {"sequence", "gram_status", "activity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return clean_mic_data(df)


def clean_mic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare MIC rows for baseline regression."""
    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["sequence", "gram_status", "activity"])
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.upper().str.strip()
    cleaned["gram_status"] = cleaned["gram_status"].astype(str).str.strip()
    cleaned["activity"] = pd.to_numeric(cleaned["activity"], errors="coerce")
    cleaned = cleaned.dropna(subset=["activity"])
    cleaned = cleaned[cleaned["gram_status"].isin(GRAM_CLASSES)]
    cleaned = cleaned[cleaned["activity"] > 0]
    cleaned = cleaned[cleaned["sequence"].str.len() > 0]
    cleaned = cleaned[~cleaned["sequence"].str.contains(NONSTANDARD_PATTERN)]
    cleaned["log_mic"] = np.log10(cleaned["activity"])
    return cleaned[["sequence", "gram_status", "activity", "log_mic"]].reset_index(
        drop=True
    )


def split_by_sequence(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> SplitData:
    """Split rows while keeping each unique sequence in exactly one split."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size, val_size, and test_size must sum to 1.0")

    splitter = GroupShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    train_idx, temp_idx = next(splitter.split(df, groups=df["sequence"]))
    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()

    relative_test_size = test_size / (val_size + test_size)
    temp_splitter = GroupShuffleSplit(
        n_splits=1, test_size=relative_test_size, random_state=random_state
    )
    val_idx, test_idx = next(temp_splitter.split(temp, groups=temp["sequence"]))

    return SplitData(
        train=train.reset_index(drop=True),
        val=temp.iloc[val_idx].reset_index(drop=True),
        test=temp.iloc[test_idx].reset_index(drop=True),
    )


def split_train_val_by_sequence(
    df: pd.DataFrame,
    train_size: float = 0.8235294117647058,
    val_size: float = 0.17647058823529413,
    random_state: int = 42,
) -> SplitData:
    """Split training rows into train/validation while grouping by sequence."""
    if not np.isclose(train_size + val_size, 1.0):
        raise ValueError("train_size and val_size must sum to 1.0")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=random_state,
    )
    train_idx, val_idx = next(splitter.split(df, groups=df["sequence"]))
    return SplitData(
        train=df.iloc[train_idx].reset_index(drop=True),
        val=df.iloc[val_idx].reset_index(drop=True),
        test=pd.DataFrame(columns=df.columns),
    )


def encode_sequences(sequences: Iterable[str]) -> pd.DataFrame:
    """Encode variable-length peptide sequences into fixed-width features."""
    rows = []
    for seq in sequences:
        normalized = str(seq).upper().strip()
        length = len(normalized)
        if length == 0:
            raise ValueError("Cannot encode an empty sequence")

        row: dict[str, float] = {"sequence_length": float(length)}
        for aa in STANDARD_AA:
            row[f"aa_frac_{aa}"] = normalized.count(aa) / length
        for name, residues in RESIDUE_GROUPS.items():
            row[name] = sum(normalized.count(aa) for aa in residues) / length
        rows.append(row)
    return pd.DataFrame(rows)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features from sequence and gram status columns."""
    sequence_features = encode_sequences(df["sequence"])
    gram_features = pd.get_dummies(df["gram_status"], prefix="gram", dtype=float)
    for col in ["gram_gram_negative", "gram_gram_positive"]:
        if col not in gram_features:
            gram_features[col] = 0.0
    gram_features = gram_features[["gram_gram_negative", "gram_gram_positive"]]
    return pd.concat(
        [
            sequence_features.reset_index(drop=True),
            gram_features.reset_index(drop=True),
        ],
        axis=1,
    )


class MicBaselineRegressor(BaseModel):
    """Random Forest baseline model for log10(MIC) regression."""

    def __init__(self, random_state: int = 42, n_estimators: int = 200):
        self.random_state = random_state
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MicBaselineRegressor":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "MicBaselineRegressor is a regression model and does not expose "
            "class probabilities."
        )


def build_model(
    random_state: int = 42, n_estimators: int = 200
) -> MicBaselineRegressor:
    """Create the baseline regressor."""
    return MicBaselineRegressor(random_state=random_state, n_estimators=n_estimators)


def evaluate_predictions(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute overall and per-gram regression metrics."""
    abs_error = np.abs(y_true - y_pred)
    log_error = abs_error
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "median_ae": float(np.median(abs_error)),
        "mean_error": float(np.mean(y_pred - y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": _safe_corr(y_true, y_pred, method="pearson"),
        "spearman": _safe_corr(y_true, y_pred, method="spearman"),
        "within_2fold": float(np.mean(log_error <= np.log10(2.0))),
        "within_4fold": float(np.mean(log_error <= np.log10(4.0))),
    }
    for gram_status in sorted(GRAM_CLASSES):
        mask = df["gram_status"].to_numpy() == gram_status
        if not mask.any():
            continue
        prefix = gram_status.replace("gram_", "")
        metrics[f"{prefix}_mae"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
        metrics[f"{prefix}_rmse"] = float(root_mean_squared_error(y_true[mask], y_pred[mask]))
    return metrics


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    """Return a finite correlation value, or 0.0 when correlation is undefined."""
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    corr = pd.Series(y_true).corr(pd.Series(y_pred), method=method)
    if pd.isna(corr):
        return 0.0
    return float(corr)


def estimator_checkpoints(n_estimators: int) -> list[int]:
    """Return increasing Random Forest checkpoints up to n_estimators."""
    checkpoints = {
        checkpoint
        for checkpoint in DEFAULT_ESTIMATOR_CHECKPOINTS
        if checkpoint <= n_estimators
    }
    checkpoints.add(n_estimators)
    return sorted(checkpoints)


def train_and_evaluate(
    input_csv: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    return_history: bool = False,
) -> dict[str, dict[str, float]] | tuple[dict[str, dict[str, float]], list[dict]]:
    """Train the baseline model and write predictions, metrics, and model file."""
    output_path = Path(output_dir)
    tables_dir = output_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_mic_data(input_csv)
    splits = split_train_val_by_sequence(df, random_state=random_state)

    split_frames = [
        ("train", splits.train),
        ("val", splits.val),
    ]

    X_train = build_features(splits.train)
    y_train = splits.train["log_mic"].to_numpy()
    model = build_model(random_state=random_state)
    total_estimators = model._model.n_estimators
    model._model.set_params(warm_start=True)

    split_features: dict[str, pd.DataFrame] = {
        "train": X_train,
    }
    split_targets: dict[str, np.ndarray] = {
        "train": y_train,
    }
    for split_name, split_df in split_frames:
        if split_df.empty or split_name == "train":
            continue
        split_features[split_name] = build_features(split_df).reindex(
            columns=X_train.columns, fill_value=0.0
        )
        split_targets[split_name] = split_df["log_mic"].to_numpy()

    metric_history: list[dict] = []
    metrics_by_split: dict[str, dict[str, float]] = {}
    for step, n_estimators in enumerate(estimator_checkpoints(total_estimators), start=1):
        model._model.set_params(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        metrics_by_split = {}
        for split_name, split_df in split_frames:
            if split_df.empty:
                continue
            y_pred = model.predict(split_features[split_name])
            metrics = evaluate_predictions(
                split_df, split_targets[split_name], y_pred
            )
            metrics_by_split[split_name] = metrics
            metric_history.append(
                {
                    "step": step,
                    "num_estimators": n_estimators,
                    "split": split_name,
                    "metrics": metrics,
                }
            )

    prediction_frames = []
    for split_name, split_df in split_frames:
        if split_df.empty:
            continue
        y_pred = model.predict(split_features[split_name])

        if split_name == "val":
            pred_df = split_df[["sequence", "gram_status", "activity", "log_mic"]].copy()
            pred_df["split"] = split_name
            pred_df["pred_log_mic"] = y_pred
            pred_df["pred_mic"] = np.power(10.0, y_pred)
            prediction_frames.append(pred_df)

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        tables_dir / "mic_baseline_predictions.csv", index=False
    )
    pd.DataFrame(metrics_by_split).T.to_csv(
        tables_dir / "mic_baseline_metrics.csv", index_label="split"
    )
    joblib.dump(
        {"model": model, "feature_columns": X_train.columns.tolist()},
        output_path / "mic_baseline_model.joblib",
    )
    if return_history:
        return metrics_by_split, metric_history
    return metrics_by_split
