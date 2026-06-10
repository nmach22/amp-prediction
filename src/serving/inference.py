"""Reproduce the training-time MIC feature pipeline for one new peptide at a time.

Loads a joblib artifact saved by ``src.models.mic_runner.train_and_evaluate_mic_baseline``
for the ``mlp_mic_physchem_esm2_pca_context_regularized`` experiment and predicts the
MIC of a single (sequence, microbe) pair without touching the training-only ESM2
embedding cache (``embeddings_for_sequences``) or the NCBI/ete3 taxonomy lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.plm import PLMEncoder
from src.models.mic_baseline import NONSTANDARD_PATTERN
from src.models.mlp_mic import _one_hot_gram_taxonomy_features, build_mlp_features

SHORT_SEQUENCE_LENGTH = 5
LONG_SEQUENCE_LENGTH = 60


@dataclass(frozen=True)
class MicrobeProfile:
    """One fixed dropdown entry's taxonomy context, matching training columns."""

    key: str
    display_name: str
    target_activity_name: str
    gram_status: str
    phylum: str
    class_: str
    order: str
    family: str
    genus: str

    def as_row(self) -> dict:
        return {
            "target_activity_name": self.target_activity_name,
            "gram_status": self.gram_status,
            "Phylum": self.phylum,
            "Class": self.class_,
            "Order": self.order,
            "Family": self.family,
            "Genus": self.genus,
        }


@dataclass
class LoadedMicModel:
    """The pieces of a trained artifact needed to score one new peptide."""

    model: object
    feature_columns: list[str]
    esm2_feature_columns: list[str]
    esm2_pca_scaler: object
    esm2_pca_model: object
    passthrough_feature_columns: list[str]


def load_mic_model(path: str | Path) -> LoadedMicModel:
    """Load a joblib bundle produced by ``train_and_evaluate_mic_baseline``."""
    bundle = joblib.load(path)
    missing = [
        key
        for key in (
            "model",
            "feature_columns",
            "esm2_feature_columns",
            "esm2_pca_scaler",
            "esm2_pca_model",
            "passthrough_feature_columns",
        )
        if key not in bundle
    ]
    if missing:
        raise ValueError(
            f"Model artifact {path} is missing required keys: {missing}. "
            "Retrain with the current pca_reduce_esm2_features, which now persists "
            "the fitted ESM2 scaler/PCA alongside the model."
        )
    return LoadedMicModel(
        model=bundle["model"],
        feature_columns=list(bundle["feature_columns"]),
        esm2_feature_columns=list(bundle["esm2_feature_columns"]),
        esm2_pca_scaler=bundle["esm2_pca_scaler"],
        esm2_pca_model=bundle["esm2_pca_model"],
        passthrough_feature_columns=list(bundle["passthrough_feature_columns"]),
    )


def load_microbes(path: str | Path) -> dict[str, MicrobeProfile]:
    """Load the fixed microbe dropdown list from its JSON lookup table."""
    import json

    records = json.loads(Path(path).read_text())
    profiles = {record["key"]: MicrobeProfile(**record) for record in records}
    if len(profiles) != len(records):
        raise ValueError(f"Duplicate microbe keys found in {path}.")
    return profiles


def validate_sequence(sequence: str) -> list[str]:
    """Return warnings for a sequence, or raise ValueError if it can't be scored."""
    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("Sequence must not be empty.")
    if NONSTANDARD_PATTERN.search(normalized):
        raise ValueError(
            "Sequence contains non-standard amino acid characters; only "
            "ACDEFGHIKLMNPQRSTVWY are supported."
        )
    warnings: list[str] = []
    if len(normalized) < SHORT_SEQUENCE_LENGTH:
        warnings.append(
            f"Sequence is unusually short (<{SHORT_SEQUENCE_LENGTH} aa); "
            "prediction may be unreliable."
        )
    if len(normalized) > LONG_SEQUENCE_LENGTH:
        warnings.append(
            f"Sequence is unusually long (>{LONG_SEQUENCE_LENGTH} aa) relative to "
            "training data; prediction may be unreliable."
        )
    return warnings


def predict_mic(
    loaded: LoadedMicModel,
    plm_encoder: PLMEncoder,
    sequence: str,
    microbe: MicrobeProfile,
) -> dict:
    """Predict log10(MIC) and MIC (ug/mL) for one new (sequence, microbe) pair."""
    normalized_sequence = sequence.strip().upper()
    row = pd.DataFrame([{"sequence": normalized_sequence, **microbe.as_row()}])

    physchem = build_mlp_features(row).add_prefix("physchem_")
    context = _one_hot_gram_taxonomy_features(row).add_prefix("context_")
    passthrough_raw = pd.concat(
        [physchem.reset_index(drop=True), context.reset_index(drop=True)], axis=1
    )
    passthrough = passthrough_raw.reindex(
        columns=loaded.passthrough_feature_columns, fill_value=0.0
    )

    embedding = plm_encoder.encode([normalized_sequence])
    esm2_raw = pd.DataFrame(
        embedding, columns=[f"esm2_{index}" for index in range(embedding.shape[1])]
    )
    esm2_raw = esm2_raw.reindex(columns=loaded.esm2_feature_columns, fill_value=0.0)

    esm2_scaled = loaded.esm2_pca_scaler.transform(esm2_raw.astype(float))
    esm2_pca = loaded.esm2_pca_model.transform(esm2_scaled)
    esm2_pca_df = pd.DataFrame(
        esm2_pca,
        columns=[f"esm2_pca_{index}" for index in range(esm2_pca.shape[1])],
    )

    combined = pd.concat(
        [esm2_pca_df.reset_index(drop=True), passthrough.reset_index(drop=True)],
        axis=1,
    )
    combined = combined.reindex(columns=loaded.feature_columns, fill_value=0.0)

    log10_mic = float(np.asarray(loaded.model.predict(combined)).ravel()[0])
    return {
        "log10_mic": log10_mic,
        "mic_ug_per_ml": float(10.0**log10_mic),
    }


__all__ = [
    "LoadedMicModel",
    "MicrobeProfile",
    "load_mic_model",
    "load_microbes",
    "predict_mic",
    "validate_sequence",
]
