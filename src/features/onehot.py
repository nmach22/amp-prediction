"""
One-hot encoding of amino acid sequences.

Each sequence is padded / truncated to `max_len` and encoded as a
(max_len × 20) binary matrix, then flattened into a 1-D feature vector.
"""

import numpy as np
from typing import List

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


class OneHotEncoder:
    """Encode peptide sequences as flattened one-hot vectors.

    Args:
        max_len: Fixed length sequences are padded/truncated to. Defaults to 50.
    """

    def __init__(self, max_len: int = 50):
        self.max_len = max_len
        self.n_features = max_len * len(AMINO_ACIDS)

    def encode(self, sequences: List[str]) -> np.ndarray:
        """Encode a list of sequences.

        Args:
            sequences: List of uppercase amino acid strings.

        Returns:
            Array of shape (n_samples, max_len * 20).
        """
        X = np.zeros((len(sequences), self.n_features), dtype=np.float32)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[: self.max_len]):
                if aa in AA_TO_IDX:
                    X[i, j * len(AMINO_ACIDS) + AA_TO_IDX[aa]] = 1.0
        return X

