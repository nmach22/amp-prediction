"""Protein Language Model (PLM) embeddings and reusable embedding caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
import numpy as np

CACHE_DIR = Path(__file__).parents[2] / "data" / "processed" / "embeddings"
DEFAULT_ESM2_MODEL = "facebook/esm2_t12_35M_UR50D"
ESM2_150M_MODEL = "facebook/esm2_t30_150M_UR50D"
DEFAULT_MIC_EMBEDDING_PATH = (
    CACHE_DIR / "facebook_esm2_t12_35M_UR50D_mic_embeddings.npz"
)
ESM2_150M_MIC_EMBEDDING_PATH = (
    CACHE_DIR / "facebook_esm2_t30_150M_UR50D_mic_embeddings.npz"
)


def model_slug(model_name: str) -> str:
    """Return a filesystem-safe model name."""
    return model_name.replace("/", "_")


def embedding_metadata_path(path: str | Path) -> Path:
    """Return the JSON metadata path for an embedding cache."""
    return Path(path).with_suffix(".json")


def save_embedding_cache(
    path: str | Path,
    sequences: list[str],
    embeddings: np.ndarray,
    *,
    model_name: str,
    metadata: dict | None = None,
) -> Path:
    """Save sequence-keyed embeddings and companion metadata."""
    path = Path(path)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(sequences) != len(embeddings):
        raise ValueError(
            "Number of sequences must match number of embedding rows: "
            f"{len(sequences)} != {len(embeddings)}"
        )
    if len(set(sequences)) != len(sequences):
        raise ValueError("Embedding cache sequences must be unique.")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sequences=np.asarray(sequences, dtype=str),
        embeddings=embeddings,
        model_name=np.asarray(model_name),
    )

    cache_metadata = {
        "model_name": model_name,
        "embedding_path": str(path),
        "sequence_count": int(len(sequences)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
    }
    if metadata:
        cache_metadata.update(metadata)
    embedding_metadata_path(path).write_text(
        json.dumps(cache_metadata, indent=2, sort_keys=True) + "\n"
    )
    return path


def load_embedding_cache(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cached sequences and embeddings from an NPZ artifact."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PLM embedding cache not found: {path}. "
            "Run scripts/make_plm_embeddings.py first."
        )
    with np.load(path, allow_pickle=False) as cache:
        sequences = cache["sequences"].astype(str)
        embeddings = cache["embeddings"].astype(np.float32, copy=False)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings in {path}, got {embeddings.shape}.")
    if len(sequences) != len(embeddings):
        raise ValueError(
            f"Cache {path} has {len(sequences)} sequences but "
            f"{len(embeddings)} embedding rows."
        )
    return sequences, embeddings


def load_embedding_cache_metadata(path: str | Path) -> dict:
    """Load companion metadata when present."""
    metadata_path = embedding_metadata_path(path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def embeddings_for_sequences(
    sequences: list[str] | np.ndarray,
    path: str | Path = DEFAULT_MIC_EMBEDDING_PATH,
) -> np.ndarray:
    """Return cached embedding rows ordered like ``sequences``."""
    cache_sequences, cache_embeddings = load_embedding_cache(path)
    index = {sequence: row for row, sequence in enumerate(cache_sequences)}
    requested = [str(sequence) for sequence in sequences]
    missing = sorted({sequence for sequence in requested if sequence not in index})
    if missing:
        examples = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} sequence(s) are missing from PLM cache {path}: "
            f"{examples}. Run scripts/make_plm_embeddings.py with the same input CSV."
        )
    return np.vstack([cache_embeddings[index[sequence]] for sequence in requested])


class PLMEncoder:
    """Extract mean-pooled residue embeddings from a pre-trained PLM.

    Args:
        model_name: HuggingFace model identifier.
        cache_dir: Directory for caching pre-computed embeddings.
            Pass None to disable caching.
        device: 'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ESM2_MODEL,
        cache_dir: Optional[Path] = CACHE_DIR,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, sequences: List[str], batch_size: int = 32) -> np.ndarray:
        """Return mean-pooled PLM embeddings for each sequence.

        Args:
            sequences: List of uppercase amino acid strings.
            batch_size: Sequences per GPU batch.

        Returns:
            Array of shape (n_samples, hidden_size).
        """
        cache_file = self._cache_path(sequences)
        if cache_file and cache_file.exists():
            return np.load(cache_file)

        embeddings = self._compute(sequences, batch_size)

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, embeddings)

        return embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load the model and tokenizer."""
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, AutoModel

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)

    def _compute(self, sequences: List[str], batch_size: int) -> np.ndarray:
        import torch

        self._load_model()
        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                return_special_tokens_mask=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            residue_mask = (
                inputs["attention_mask"].bool()
                & ~inputs["special_tokens_mask"].bool()
            )
            attention_mask = residue_mask.unsqueeze(-1).float()
            token_embeddings = outputs.last_hidden_state
            denominator = attention_mask.sum(1).clamp(min=1.0)
            mean_embeddings = (token_embeddings * attention_mask).sum(1) / denominator
            all_embeddings.append(mean_embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def _cache_path(self, sequences: List[str]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        import hashlib
        key = hashlib.md5("".join(sequences).encode()).hexdigest()[:12]
        return self.cache_dir / f"{model_slug(self.model_name)}_{key}.npy"
