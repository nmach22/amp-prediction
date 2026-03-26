"""
Physicochemical descriptor encoding.

For each sequence, computes a fixed-size vector of global descriptors:
  - Amino acid composition (20 values)
  - Molecular weight
  - Isoelectric point
  - Aromaticity
  - Instability index
  - Gravy (hydrophobicity index)
  - Secondary structure fractions (helix, turn, sheet)

Requires: biopython
"""

from typing import List
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class PhysicochemicalEncoder:
    """Encode peptide sequences using global physicochemical descriptors."""

    # Feature names, useful for logging / feature importance plots
    FEATURE_NAMES = (
        [f"aa_comp_{aa}" for aa in list("ACDEFGHIKLMNPQRSTVWY")]
        + [
            "molecular_weight",
            "isoelectric_point",
            "aromaticity",
            "instability_index",
            "gravy",
            "helix_fraction",
            "turn_fraction",
            "sheet_fraction",
        ]
    )

    def encode(self, sequences: List[str]) -> np.ndarray:
        """Encode a list of sequences.

        Args:
            sequences: List of uppercase amino acid strings.

        Returns:
            Array of shape (n_samples, 28).
        """
        vectors = [self._encode_one(s) for s in sequences]
        return np.array(vectors, dtype=np.float32)

    def _encode_one(self, seq: str) -> List[float]:
        analysis = ProteinAnalysis(seq)
        aa_comp = list(analysis.get_amino_acids_percent().values())
        helix, turn, sheet = analysis.secondary_structure_fraction()
        return aa_comp + [
            analysis.molecular_weight(),
            analysis.isoelectric_point(),
            analysis.aromaticity(),
            analysis.instability_index(),
            analysis.gravy(),
            helix,
            turn,
            sheet,
        ]

