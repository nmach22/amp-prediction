"""
Word2Vec (gensim) sequence embeddings.

Sequences are tokenised into overlapping k-mer "words". Each sequence is
represented as the mean of its token vectors (bag-of-words mean pooling).

Usage:
    encoder = Word2VecEncoder(k=3, vector_size=100)
    encoder.fit(train_sequences)          # train the word2vec model
    X_train = encoder.encode(train_sequences)
    X_test  = encoder.encode(test_sequences)
"""

from typing import List, Optional
import numpy as np
from gensim.models import Word2Vec


def _kmerize(sequence: str, k: int) -> List[str]:
    """Split a sequence into overlapping k-mers."""
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


class Word2VecEncoder:
    """Encode peptide sequences using a trained Word2Vec model.

    Args:
        k: k-mer size (default 3).
        vector_size: Embedding dimensionality (default 100).
        window: Word2Vec context window (default 5).
        min_count: Minimum token frequency (default 1).
        epochs: Training epochs (default 10).
        workers: Parallel workers (default 4).
    """

    def __init__(
        self,
        k: int = 3,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 10,
        workers: int = 4,
    ):
        self.k = k
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.model: Optional[Word2Vec] = None

    def fit(self, sequences: List[str]) -> "Word2VecEncoder":
        """Train the Word2Vec model on the given sequences.

        Args:
            sequences: Training sequences (uppercase amino acid strings).

        Returns:
            self
        """
        corpus = [_kmerize(s, self.k) for s in sequences]
        self.model = Word2Vec(
            sentences=corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=self.workers,
        )
        return self

    def encode(self, sequences: List[str]) -> np.ndarray:
        """Mean-pool k-mer embeddings for each sequence.

        Args:
            sequences: List of uppercase amino acid strings.

        Returns:
            Array of shape (n_samples, vector_size).
        """
        if self.model is None:
            raise RuntimeError("Call fit() before encode().")
        X = np.zeros((len(sequences), self.vector_size), dtype=np.float32)
        for i, seq in enumerate(sequences):
            kmers = _kmerize(seq, self.k)
            vecs = [
                self.model.wv[km]
                for km in kmers
                if km in self.model.wv
            ]
            if vecs:
                X[i] = np.mean(vecs, axis=0)
        return X

