"""
Genome-level target encoding for bacterial organisms.

Provides feature vectors based on:
- Oligonucleotide composition (k=3, k=4, k=5 normalized frequencies)
- Inter-genome similarity: dDDH estimates and gyrB pairwise identity

Features are cached as parquet files. The encoder maps target_activity_name
to a fixed-width numeric vector suitable for regression models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_GENOME_DIR = Path("data/processed/embeddings/genome")
DEFAULT_MAPPING_PATH = DEFAULT_GENOME_DIR / "strain_assembly_map.csv"
DEFAULT_OLIGO_PATH = DEFAULT_GENOME_DIR / "oligo_composition.parquet"
DEFAULT_DDH_PATH = DEFAULT_GENOME_DIR / "ddh_estimates.parquet"
DEFAULT_GYRB_PATH = DEFAULT_GENOME_DIR / "gyrb_similarity.parquet"

# Landmark species for similarity encoding (most common in AMP datasets)
DEFAULT_LANDMARKS = [
    "Escherichia coli",
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Bacillus subtilis",
    "Salmonella enterica",
    "Staphylococcus epidermidis",
    "Enterococcus faecalis",
    "Enterococcus faecium",
]


def _extract_species(name: object) -> str:
    """Extract binomial name from target_activity_name."""
    words = str(name).strip().split()
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    return str(name).strip()


class GenomeEncoder:
    """
    Encode target organisms using genome-level features.

    Features include:
    - Oligonucleotide frequencies (k=3,4,5): 64 + 256 + 1024 = 1344 features
    - dDDH similarity to landmark species: N_landmarks features
    - gyrB similarity to landmark species: N_landmarks features

    Total: 1344 + 2 * N_landmarks (default: 1344 + 20 = 1364 features)
    """

    def __init__(
        self,
        genome_dir: str | Path = DEFAULT_GENOME_DIR,
        oligo_path: str | Path | None = None,
        ddh_path: str | Path | None = None,
        gyrb_path: str | Path | None = None,
        mapping_path: str | Path | None = None,
        landmarks: list[str] | None = None,
        n_pca_components: int | None = None,
    ):
        self.genome_dir = Path(genome_dir)
        self.oligo_path = Path(oligo_path) if oligo_path else DEFAULT_OLIGO_PATH
        self.ddh_path = Path(ddh_path) if ddh_path else DEFAULT_DDH_PATH
        self.gyrb_path = Path(gyrb_path) if gyrb_path else DEFAULT_GYRB_PATH
        self.mapping_path = Path(mapping_path) if mapping_path else DEFAULT_MAPPING_PATH
        self.landmarks = landmarks or DEFAULT_LANDMARKS
        self.n_pca_components = n_pca_components

        # Loaded data (lazy)
        self._oligo_df: Optional[pd.DataFrame] = None
        self._ddh_df: Optional[pd.DataFrame] = None
        self._gyrb_df: Optional[pd.DataFrame] = None
        self._mapping_df: Optional[pd.DataFrame] = None

        # Fitted transformers
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._fitted = False

    def _load_oligo(self) -> pd.DataFrame:
        if self._oligo_df is None:
            if self.oligo_path.exists():
                self._oligo_df = pd.read_parquet(self.oligo_path)
            else:
                self._oligo_df = pd.DataFrame()
        return self._oligo_df

    def _load_ddh(self) -> pd.DataFrame:
        if self._ddh_df is None:
            if self.ddh_path.exists():
                self._ddh_df = pd.read_parquet(self.ddh_path)
            else:
                self._ddh_df = pd.DataFrame()
        return self._ddh_df

    def _load_gyrb(self) -> pd.DataFrame:
        if self._gyrb_df is None:
            if self.gyrb_path.exists():
                self._gyrb_df = pd.read_parquet(self.gyrb_path)
            else:
                self._gyrb_df = pd.DataFrame()
        return self._gyrb_df

    def _oligo_vector(self, species: str) -> np.ndarray:
        """Get oligonucleotide frequency vector for a species."""
        oligo_df = self._load_oligo()
        if oligo_df.empty or "species" not in oligo_df.columns:
            return np.zeros(1344, dtype=np.float32)

        match = oligo_df[oligo_df["species"] == species]
        if match.empty:
            return np.zeros(1344, dtype=np.float32)

        # All columns except 'species' are features
        feature_cols = [c for c in oligo_df.columns if c != "species"]
        return match[feature_cols].values[0].astype(np.float32)

    def _ddh_similarity_vector(self, species: str) -> np.ndarray:
        """Get dDDH similarity to landmark species."""
        ddh_df = self._load_ddh()
        n_landmarks = len(self.landmarks)

        if ddh_df.empty or species not in ddh_df.index:
            return np.zeros(n_landmarks, dtype=np.float32)

        # Get similarity to each landmark
        vector = np.zeros(n_landmarks, dtype=np.float32)
        for i, landmark in enumerate(self.landmarks):
            if landmark in ddh_df.columns:
                vector[i] = ddh_df.loc[species, landmark]
        return vector

    def _gyrb_similarity_vector(self, species: str) -> np.ndarray:
        """Get gyrB similarity to landmark species."""
        gyrb_df = self._load_gyrb()
        n_landmarks = len(self.landmarks)

        if gyrb_df.empty or species not in gyrb_df.index:
            return np.zeros(n_landmarks, dtype=np.float32)

        vector = np.zeros(n_landmarks, dtype=np.float32)
        for i, landmark in enumerate(self.landmarks):
            if landmark in gyrb_df.columns:
                vector[i] = gyrb_df.loc[species, landmark]
        return vector

    def encode_single(self, target_name: object) -> np.ndarray:
        """Encode a single target organism name into a genome feature vector."""
        species = _extract_species(target_name)

        oligo = self._oligo_vector(species)
        ddh = self._ddh_similarity_vector(species)
        gyrb = self._gyrb_similarity_vector(species)

        return np.concatenate([oligo, ddh, gyrb])

    def encode(self, target_names: pd.Series | list) -> np.ndarray:
        """Encode a series of target organism names into a feature matrix."""
        if isinstance(target_names, list):
            target_names = pd.Series(target_names)

        vectors = np.stack([self.encode_single(name) for name in target_names])
        return vectors

    def fit(self, target_names: pd.Series | list) -> "GenomeEncoder":
        """Fit scaler (and optional PCA) on training data."""
        raw = self.encode(target_names)

        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(raw)

        if self.n_pca_components is not None:
            self._pca = PCA(n_components=self.n_pca_components)
            self._pca.fit(scaled)

        self._fitted = True
        return self

    def transform(self, target_names: pd.Series | list) -> np.ndarray:
        """Transform target names using fitted scaler/PCA."""
        raw = self.encode(target_names)

        if self._scaler is not None:
            result = self._scaler.transform(raw)
        else:
            result = raw

        if self._pca is not None:
            result = self._pca.transform(result)

        return result

    def fit_transform(self, target_names: pd.Series | list) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(target_names)
        return self.transform(target_names)

    @property
    def feature_dim(self) -> int:
        """Return the output feature dimension."""
        if self.n_pca_components is not None:
            return self.n_pca_components
        n_landmarks = len(self.landmarks)
        return 1344 + 2 * n_landmarks  # oligo + ddh + gyrb

    def feature_names(self) -> list[str]:
        """Return feature column names (before PCA)."""
        import itertools

        names = []
        for k in (3, 4, 5):
            bases = "ACGT"
            for combo in itertools.product(bases, repeat=k):
                names.append(f"kmer_{k}_{''.join(combo)}")

        for landmark in self.landmarks:
            names.append(f"ddh_{landmark.replace(' ', '_')}")
        for landmark in self.landmarks:
            names.append(f"gyrb_{landmark.replace(' ', '_')}")

        return names
