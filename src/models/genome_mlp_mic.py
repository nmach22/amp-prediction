"""
Genome-encoded MLP MIC regression model.

Trains an MLP regressor on genome-level target features (oligonucleotide
composition, dDDH, gyrB similarity) for log10(MIC) prediction.

Supports two ablation modes:
- genome_only: genome features as sole input
- combined: genome features concatenated with peptide embeddings (ESM-2)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.features.genome import GenomeEncoder
from src.features.plm import (
    DEFAULT_ESM2_MODEL,
    DEFAULT_MIC_EMBEDDING_PATH,
    embeddings_for_sequences,
)
from src.models.mlp_mic import MlpMicRegressor
from src.models.taxonomy_mic_baseline import evaluate_taxonomy_predictions


ABLATION_MODES = ("genome_only", "combined")


def load_genome_mic_data(path: str) -> pd.DataFrame:
    """Load and clean MIC rows — only requires sequence, target_activity_name, activity."""
    import re
    df = pd.read_csv(path)
    required = {"sequence", "target_activity_name", "activity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["sequence", "target_activity_name", "activity"])
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.upper().str.strip()
    cleaned["target_activity_name"] = cleaned["target_activity_name"].astype(str).str.strip()
    cleaned["activity"] = pd.to_numeric(cleaned["activity"], errors="coerce")
    cleaned = cleaned.dropna(subset=["activity"])
    cleaned = cleaned[cleaned["activity"] > 0]
    cleaned = cleaned[cleaned["sequence"].str.len() > 0]
    # Remove sequences with non-standard amino acids
    cleaned = cleaned[~cleaned["sequence"].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", flags=re.IGNORECASE)]
    cleaned["log_mic"] = np.log10(cleaned["activity"])
    return cleaned.reset_index(drop=True)


def build_genome_features(
    df: pd.DataFrame,
    genome_encoder: GenomeEncoder,
    fitted: bool = False,
) -> pd.DataFrame:
    """Build genome-only feature matrix from target_activity_name."""
    target_names = df["target_activity_name"]

    if not fitted:
        features = genome_encoder.fit_transform(target_names)
    else:
        features = genome_encoder.transform(target_names)

    feature_names = genome_encoder.feature_names()
    if features.shape[1] != len(feature_names):
        # PCA was applied, use generic names
        feature_names = [f"genome_pca_{i}" for i in range(features.shape[1])]

    return pd.DataFrame(features, columns=feature_names, index=df.index)


def build_combined_features(
    df: pd.DataFrame,
    genome_encoder: GenomeEncoder,
    fitted: bool = False,
) -> pd.DataFrame:
    """Build combined genome + peptide (ESM-2) feature matrix."""
    # Genome features
    genome_feats = build_genome_features(df, genome_encoder, fitted=fitted)

    # Peptide features: ESM-2 embeddings
    sequences = df["sequence"].astype(str).tolist()
    embeddings = embeddings_for_sequences(sequences, DEFAULT_MIC_EMBEDDING_PATH)
    esm2_feats = pd.DataFrame(
        embeddings,
        columns=[f"esm2_{i}" for i in range(embeddings.shape[1])],
        index=df.index,
    )

    return pd.concat(
        [genome_feats.reset_index(drop=True), esm2_feats.reset_index(drop=True)],
        axis=1,
    ).astype(float)


class GenomeMlpMicRegressor(MlpMicRegressor):
    """
    MLP MIC regressor using genome-level target encoding.

    Feature construction is handled externally by build_genome_features() or
    build_combined_features() (called via the registry's build_features spec).
    This class just wraps MlpMicRegressor with genome-specific hyperparameter
    defaults and metadata.

    Parameters
    ----------
    mode : str
        Ablation mode label: 'genome_only' or 'combined' (genome + ESM-2).
    genome_dir : str
        Path to cached genome features directory (for metadata only).
    n_pca_components : int or None
        If set, apply PCA to genome features before the MLP.
    landmarks : list of str or None
        Landmark species for similarity encoding.
    """

    def __init__(
        self,
        mode: str = "genome_only",
        genome_dir: str = "data/processed/embeddings/genome",
        n_pca_components: int | None = None,
        landmarks: list[str] | None = None,
        random_state: int = 42,
        hidden_layers: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.25,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 400,
        patience: int = 40,
        batch_size: int = 64,
        noise_std: float = 0.01,
    ):
        if mode not in ABLATION_MODES:
            raise ValueError(f"mode must be one of {ABLATION_MODES}, got '{mode}'")

        super().__init__(
            random_state=random_state,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            noise_std=noise_std,
        )
        self.mode = mode
        self.genome_dir = genome_dir
        self.n_pca_components = n_pca_components
        self.landmarks = landmarks

    def artifact_metadata(self, feature_columns: list[str] | None = None) -> dict:
        """Return model metadata for logging."""
        meta = super().artifact_metadata(feature_columns or [])
        meta.update({
            "genome_mode": self.mode,
            "genome_n_pca_components": self.n_pca_components,
            "genome_landmarks": self.landmarks,
        })
        return meta


def run_genome_mlp_experiment(
    train_path: str,
    test_path: str | None = None,
    mode: str = "genome_only",
    genome_dir: str = "data/processed/embeddings/genome",
    n_pca_components: int | None = None,
    random_state: int = 42,
    hidden_layers: tuple[int, ...] = (512, 256, 128),
    dropout: float = 0.25,
    learning_rate: float = 5e-4,
    max_epochs: int = 400,
    patience: int = 40,
    batch_size: int = 64,
) -> dict:
    """
    Run the genome MLP MIC regression experiment.

    Returns a dict with train/test metrics and model metadata.
    """
    # Load data
    train_df = load_genome_mic_data(train_path)
    test_df = load_genome_mic_data(test_path) if test_path else None

    y_train = train_df["activity"].values
    y_test = test_df["activity"].values if test_df is not None else None

    # Build model
    model = GenomeMlpMicRegressor(
        mode=mode,
        genome_dir=genome_dir,
        n_pca_components=n_pca_components,
        random_state=random_state,
        hidden_layers=hidden_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
    )

    # Fit
    model.fit(
        train_df,
        y_train,
        X_val=test_df,
        y_val=y_test,
    )

    # Evaluate
    results = {"mode": mode, "model": model}

    train_preds = model.predict(train_df)
    results["train_metrics"] = evaluate_taxonomy_predictions(y_train, train_preds)

    if test_df is not None and y_test is not None:
        test_preds = model.predict(test_df)
        results["test_metrics"] = evaluate_taxonomy_predictions(y_test, test_preds)

    results["metadata"] = model.artifact_metadata()
    return results
