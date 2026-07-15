"""Inference helpers for saved MIC regression bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from src.models.mlp_mic import (
    build_mlp_esm2_context_features,
    build_mlp_features,
    build_mlp_physchem_esm2_context_features,
)

MIC_INFERENCE_SCHEMA_VERSION = "mic-inference-bundle-v1"

FEATURE_BUILDERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "mlp_physchem": build_mlp_features,
    "mlp_esm2_context": build_mlp_esm2_context_features,
    "mlp_physchem_esm2_context": build_mlp_physchem_esm2_context_features,
}

REQUIRED_INPUT_COLUMNS: dict[str, tuple[str, ...]] = {
    "mlp_physchem": ("sequence",),
    "mlp_esm2_context": ("sequence",),
    "mlp_physchem_esm2_context": ("sequence",),
}


def load_mic_inference_bundle(path: str | Path) -> dict:
    """Load and validate a saved MIC inference bundle."""
    bundle = joblib.load(path)
    schema_version = bundle.get("schema_version")
    if schema_version != MIC_INFERENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported MIC inference bundle schema {schema_version!r}; "
            f"expected {MIC_INFERENCE_SCHEMA_VERSION!r}."
        )
    missing = {"model", "feature_columns", "feature_builder"} - set(bundle)
    if missing:
        raise ValueError(f"MIC inference bundle is missing keys: {sorted(missing)}")
    feature_builder_name = bundle["feature_builder"]
    if feature_builder_name not in FEATURE_BUILDERS:
        raise ValueError(f"Unknown MIC feature builder: {feature_builder_name!r}")
    return bundle


def predict_mic_dataframe(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Predict log10(MIC) and MIC values for raw peptide/context rows."""
    feature_builder_name = bundle["feature_builder"]
    required_columns = REQUIRED_INPUT_COLUMNS[feature_builder_name]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing columns: {missing_columns}")

    feature_builder = FEATURE_BUILDERS[feature_builder_name]
    features = feature_builder(df).reindex(
        columns=bundle["feature_columns"],
        fill_value=0.0,
    )
    pred_log_mic = bundle["model"].predict(features)

    output = df.copy()
    output["pred_log_mic"] = pred_log_mic
    output["pred_mic"] = np.power(10.0, pred_log_mic)
    return output


def predict_mic_csv(
    bundle_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Run MIC inference from CSV input and save predictions."""
    bundle = load_mic_inference_bundle(bundle_path)
    df = pd.read_csv(input_csv)
    predictions = predict_mic_dataframe(bundle, df)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions
