"""Write a structurally-valid but non-predictive MIC model artifact.

Lets the serving app (app/main.py) start up and be demoed locally before the real
mlp_mic_physchem_esm2_pca_context_regularized artifact has been trained. Defined as
a real module (not inline in a -c script) so joblib can unpickle the placeholder
model class from a different process (e.g. uvicorn).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ESM2_DIM = 480
OUTPUT_PATH = (
    ROOT
    / "results"
    / "models"
    / "mlp_mic_physchem_esm2_pca_context_regularized_model.joblib"
)


class PlaceholderModel:
    """Deterministic stand-in for the real MLP; NOT a real prediction."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.tanh(X.to_numpy().mean(axis=1)) - 2.0


def build_placeholder_artifact() -> dict:
    rng = np.random.default_rng(0)
    esm2_columns = [f"esm2_{i}" for i in range(ESM2_DIM)]
    train_esm2 = pd.DataFrame(
        rng.normal(size=(200, ESM2_DIM)), columns=esm2_columns
    )
    scaler = StandardScaler().fit(train_esm2)
    pca = PCA(n_components=32, random_state=42).fit(scaler.transform(train_esm2))

    passthrough_feature_columns = [
        "physchem_target_activity_name_Escherichia coli ATCC 25922",
        "physchem_target_activity_name_Staphylococcus aureus ATCC 25923",
        "physchem_target_activity_name_Pseudomonas aeruginosa ATCC 27853",
        "physchem_target_activity_name_Acinetobacter baumannii ATCC 19606",
        "physchem_target_activity_name_Klebsiella pneumoniae ATCC 700603",
        "physchem_target_activity_name_Enterococcus faecalis ATCC 29212",
        "context_gram_status_gram_negative",
        "context_gram_status_gram_positive",
    ]
    return {
        "model": PlaceholderModel(),
        "feature_columns": [f"esm2_pca_{i}" for i in range(32)]
        + passthrough_feature_columns,
        "esm2_feature_columns": esm2_columns,
        "esm2_pca_scaler": scaler,
        "esm2_pca_model": pca,
        "passthrough_feature_columns": passthrough_feature_columns,
        "placeholder": True,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(build_placeholder_artifact(), OUTPUT_PATH)
    print(f"Wrote placeholder artifact to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
