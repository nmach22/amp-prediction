"""Fixed-width peptide descriptors for MIC regression models."""

from __future__ import annotations

from collections.abc import Iterable
import re

import numpy as np
import pandas as pd
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor

STANDARD_AA = tuple("ACDEFGHIKLMNPQRSTVWY")
NONSTANDARD_PATTERN = re.compile(f"[^{''.join(STANDARD_AA)}]")


class SequenceDescriptorEncoder:
    """Encode peptide sequences using AMP-relevant physicochemical descriptors."""

    SCALE_COLUMNS = (
        "eisenberg_global",
        "eisenberg_moment",
        "gravy_global",
        "gravy_moment",
        "aasi_global",
        "aasi_moment",
        "charge_phys_global",
        "charge_phys_moment",
    )

    def __init__(self, pepcats_window: int = 7):
        self.pepcats_window = pepcats_window

    def encode(self, sequences: Iterable[str]) -> pd.DataFrame:
        """Return numeric descriptors for standard amino-acid sequences."""
        normalized = [self._normalize_sequence(seq) for seq in sequences]
        if not normalized:
            return pd.DataFrame(columns=self.feature_names())

        return pd.concat(
            [
                self._global_descriptors(normalized),
                self._aa_composition(normalized),
                self._scale_descriptors(normalized),
                self._z3_autocorr(normalized),
                self._pepcats_autocorr(normalized),
            ],
            axis=1,
        ).astype(float)

    def feature_names(self) -> list[str]:
        """Return the expected descriptor column order."""
        return (
            [
                "modlamp_length",
                "modlamp_mw",
                "modlamp_charge",
                "modlamp_charge_density",
                "modlamp_pi",
                "modlamp_instability",
                "modlamp_aromaticity",
                "modlamp_aliphatic",
                "modlamp_boman",
                "modlamp_hydrophobic_ratio",
            ]
            + [f"aa_frac_{aa}" for aa in STANDARD_AA]
            + list(self.SCALE_COLUMNS)
            + [f"z3_autocorr_lag1_dim{i}" for i in range(1, 4)]
            + [
                f"pepcats_autocorr_lag{lag}_dim{dim}"
                for lag in range(1, self.pepcats_window + 1)
                for dim in range(1, 7)
            ]
            + ["pepcats_available"]
        )

    def _normalize_sequence(self, sequence: str) -> str:
        normalized = str(sequence).upper().strip()
        if not normalized:
            raise ValueError("Cannot encode an empty sequence")
        if NONSTANDARD_PATTERN.search(normalized):
            raise ValueError(f"Sequence contains nonstandard amino acids: {sequence!r}")
        return normalized

    def _global_descriptors(self, sequences: list[str]) -> pd.DataFrame:
        descriptor = GlobalDescriptor(sequences)
        descriptor.calculate_all(amide=True)
        return pd.DataFrame(
            descriptor.descriptor,
            columns=[
                "modlamp_length",
                "modlamp_mw",
                "modlamp_charge",
                "modlamp_charge_density",
                "modlamp_pi",
                "modlamp_instability",
                "modlamp_aromaticity",
                "modlamp_aliphatic",
                "modlamp_boman",
                "modlamp_hydrophobic_ratio",
            ],
        )

    def _aa_composition(self, sequences: list[str]) -> pd.DataFrame:
        rows = []
        for seq in sequences:
            length = len(seq)
            rows.append({f"aa_frac_{aa}": seq.count(aa) / length for aa in STANDARD_AA})
        return pd.DataFrame(rows)

    def _scale_descriptors(self, sequences: list[str]) -> pd.DataFrame:
        parts = []
        for scale_name, column_prefix in [
            ("eisenberg", "eisenberg"),
            ("gravy", "gravy"),
            ("aasi", "aasi"),
            ("charge_phys", "charge_phys"),
        ]:
            descriptor = PeptideDescriptor(sequences, scale_name)
            descriptor.calculate_global()
            descriptor.calculate_moment(append=True)
            parts.append(
                pd.DataFrame(
                    descriptor.descriptor,
                    columns=[
                        f"{column_prefix}_global",
                        f"{column_prefix}_moment",
                    ],
                )
            )
        return pd.concat(parts, axis=1)

    def _z3_autocorr(self, sequences: list[str]) -> pd.DataFrame:
        scale = PeptideDescriptor(["A"], "z3").scale
        rows = []
        for seq in sequences:
            rows.append(
                {
                    f"z3_autocorr_lag1_dim{dim + 1}": self._autocorr_value(
                        seq, scale, lag=1, dim=dim
                    )
                    for dim in range(3)
                }
            )
        return pd.DataFrame(rows)

    def _pepcats_autocorr(self, sequences: list[str]) -> pd.DataFrame:
        scale = PeptideDescriptor(["A"], "pepcats").scale
        rows = []
        for seq in sequences:
            row = {}
            for lag in range(1, self.pepcats_window + 1):
                for dim in range(6):
                    row[f"pepcats_autocorr_lag{lag}_dim{dim + 1}"] = (
                        self._autocorr_value(seq, scale, lag=lag, dim=dim)
                    )
            row["pepcats_available"] = float(len(seq) > self.pepcats_window)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _autocorr_value(
        sequence: str,
        scale: dict[str, list[float]],
        lag: int,
        dim: int,
    ) -> float:
        if len(sequence) <= lag:
            return 0.0
        values = np.array([scale[aa][dim] for aa in sequence], dtype=float)
        return float(np.mean(values[:-lag] * values[lag:]))
