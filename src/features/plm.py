"""
Protein Language Model (PLM) embeddings.

Supports:
  - ESM-2  (facebook/esm2_t6_8M_UR50D  or any ESM-2 variant)
  - ProtT5 (Rostlab/prot_t5_xl_half_uniref50-enc) – TODO

Pre-computed embeddings are cached to disk so GPU is only used once.
Set cache_dir=None to disable caching (not recommended).

NOTE: This module is a stub — fill in the model variant you decide to use.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import numpy as np

CACHE_DIR = Path(__file__).parents[2] / "data" / "processed" / "embeddings"


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
        model_name: str = "facebook/esm2_t6_8M_UR50D",
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
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Mean-pool over the sequence length dimension (ignore padding)
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            token_embeddings = outputs.last_hidden_state
            mean_embeddings = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
            all_embeddings.append(mean_embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def _cache_path(self, sequences: List[str]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        import hashlib
        key = hashlib.md5("".join(sequences).encode()).hexdigest()[:12]
        model_slug = self.model_name.replace("/", "_")
        return self.cache_dir / f"{model_slug}_{key}.npy"

