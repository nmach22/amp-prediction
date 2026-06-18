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

    FEATURE_SETS = {
        "full_modlamp",
        "basic",
        "composition",
        "amp_core",
        "interaction_core",
        "motif_core",
    }
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

    GROUP_COLUMNS = tuple(f"aa_{name}" for name in [
        "frac_positive",
        "frac_negative",
        "frac_polar",
        "frac_hydrophobic",
        "frac_aromatic",
    ])

    BASIC_COLUMNS = (
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
    )
    AMP_CORE_COLUMNS = (
        "modlamp_length",
        "modlamp_charge",
        "modlamp_charge_density",
        "modlamp_aromaticity",
        "modlamp_boman",
        "modlamp_hydrophobic_ratio",
        "eisenberg_global",
        "eisenberg_moment",
        "gravy_global",
        "gravy_moment",
        "charge_phys_global",
        "charge_phys_moment",
        "aa_frac_K",
        "aa_frac_R",
        "aa_frac_D",
        "aa_frac_E",
        "aa_frac_W",
        "aa_frac_F",
        "aa_frac_Y",
        "aa_frac_C",
        *GROUP_COLUMNS,
    )
    LOCAL_COLUMNS = (
        "local_max_positive_frac_w5",
        "local_max_negative_frac_w5",
        "local_max_hydrophobic_frac_w5",
        "local_max_aromatic_frac_w5",
        "local_max_charge_balance_w5",
        "longest_positive_run",
        "longest_hydrophobic_run",
        "n_terminal_positive_frac",
        "n_terminal_hydrophobic_frac",
        "c_terminal_positive_frac",
        "c_terminal_hydrophobic_frac",
    )
    REDUCED_ALPHABET = {
        "K": "pos",
        "R": "pos",
        "H": "pos",
        "D": "neg",
        "E": "neg",
        "A": "hyd",
        "I": "hyd",
        "L": "hyd",
        "M": "hyd",
        "V": "hyd",
        "F": "aro",
        "W": "aro",
        "Y": "aro",
        "S": "pol",
        "T": "pol",
        "N": "pol",
        "Q": "pol",
        "C": "pol",
        "G": "gly",
        "P": "pro",
    }
    REDUCED_SYMBOLS = ("pos", "neg", "hyd", "aro", "pol", "gly", "pro")

    def __init__(self, pepcats_window: int = 7, feature_set: str = "full_modlamp"):
        if feature_set not in self.FEATURE_SETS:
            available = ", ".join(sorted(self.FEATURE_SETS))
            raise ValueError(f"Unknown feature_set '{feature_set}'. Available: {available}")
        self.pepcats_window = pepcats_window
        self.feature_set = feature_set

    def encode(self, sequences: Iterable[str]) -> pd.DataFrame:
        """Return numeric descriptors for standard amino-acid sequences."""
        normalized = [self._normalize_sequence(seq) for seq in sequences]
        if not normalized:
            return pd.DataFrame(columns=self.feature_names())

        parts = [
            self._global_descriptors(normalized),
            self._aa_composition(normalized),
            self._group_composition(normalized),
            self._local_pattern_features(normalized),
            self._scale_descriptors(normalized),
            self._z3_autocorr(normalized),
            self._pepcats_autocorr(normalized),
        ]
        if self.feature_set == "motif_core":
            parts.append(self._reduced_kmer_composition(normalized))
        features = pd.concat(parts, axis=1).astype(float)
        return features[self.feature_names()]

    def feature_names(self) -> list[str]:
        """Return the expected descriptor column order."""
        full_columns = (
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
            + list(self.GROUP_COLUMNS)
            + list(self.LOCAL_COLUMNS)
            + list(self.SCALE_COLUMNS)
            + [f"z3_autocorr_lag1_dim{i}" for i in range(1, 4)]
            + [
                f"pepcats_autocorr_lag{lag}_dim{dim}"
                for lag in range(1, self.pepcats_window + 1)
                for dim in range(1, 7)
            ]
            + ["pepcats_available"]
        )
        if self.feature_set == "full_modlamp":
            return full_columns
        if self.feature_set == "basic":
            return list(self.BASIC_COLUMNS)
        if self.feature_set == "composition":
            return [f"aa_frac_{aa}" for aa in STANDARD_AA] + list(self.GROUP_COLUMNS)
        if self.feature_set == "interaction_core":
            return list(
                dict.fromkeys(
                    [
                        *self.AMP_CORE_COLUMNS,
                        *self.LOCAL_COLUMNS,
                        "modlamp_mw",
                        "modlamp_pi",
                        "modlamp_instability",
                        "aa_frac_G",
                        "aa_frac_P",
                        "aa_frac_H",
                    ]
                )
            )
        if self.feature_set == "motif_core":
            return full_columns + self._reduced_kmer_feature_names()
        return list(self.AMP_CORE_COLUMNS)

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

    def _group_composition(self, sequences: list[str]) -> pd.DataFrame:
        residue_groups = {
            "aa_frac_positive": set("KRH"),
            "aa_frac_negative": set("DE"),
            "aa_frac_polar": set("STNQCY"),
            "aa_frac_hydrophobic": set("AILMFWV"),
            "aa_frac_aromatic": set("FWY"),
        }
        rows = []
        for seq in sequences:
            length = len(seq)
            rows.append(
                {
                    name: sum(seq.count(aa) for aa in residues) / length
                    for name, residues in residue_groups.items()
                }
            )
        return pd.DataFrame(rows)

    def _local_pattern_features(self, sequences: list[str]) -> pd.DataFrame:
        positive = set("KRH")
        negative = set("DE")
        hydrophobic = set("AILMFWV")
        aromatic = set("FWY")
        rows = []
        for seq in sequences:
            rows.append(
                {
                    "local_max_positive_frac_w5": self._max_window_fraction(
                        seq, positive, window=5
                    ),
                    "local_max_negative_frac_w5": self._max_window_fraction(
                        seq, negative, window=5
                    ),
                    "local_max_hydrophobic_frac_w5": self._max_window_fraction(
                        seq, hydrophobic, window=5
                    ),
                    "local_max_aromatic_frac_w5": self._max_window_fraction(
                        seq, aromatic, window=5
                    ),
                    "local_max_charge_balance_w5": self._max_window_charge_balance(
                        seq, positive, negative, window=5
                    ),
                    "longest_positive_run": self._longest_run(seq, positive),
                    "longest_hydrophobic_run": self._longest_run(seq, hydrophobic),
                    "n_terminal_positive_frac": self._terminal_fraction(
                        seq, positive, terminal="n", width=5
                    ),
                    "n_terminal_hydrophobic_frac": self._terminal_fraction(
                        seq, hydrophobic, terminal="n", width=5
                    ),
                    "c_terminal_positive_frac": self._terminal_fraction(
                        seq, positive, terminal="c", width=5
                    ),
                    "c_terminal_hydrophobic_frac": self._terminal_fraction(
                        seq, hydrophobic, terminal="c", width=5
                    ),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _max_window_fraction(sequence: str, residues: set[str], window: int) -> float:
        if len(sequence) <= window:
            return sum(aa in residues for aa in sequence) / len(sequence)
        return max(
            sum(aa in residues for aa in sequence[i : i + window]) / window
            for i in range(len(sequence) - window + 1)
        )

    @staticmethod
    def _max_window_charge_balance(
        sequence: str,
        positive: set[str],
        negative: set[str],
        window: int,
    ) -> float:
        if len(sequence) <= window:
            window_seq = sequence
            denominator = len(sequence)
            return (
                sum(aa in positive for aa in window_seq)
                - sum(aa in negative for aa in window_seq)
            ) / denominator
        return max(
            (
                sum(aa in positive for aa in sequence[i : i + window])
                - sum(aa in negative for aa in sequence[i : i + window])
            )
            / window
            for i in range(len(sequence) - window + 1)
        )

    @staticmethod
    def _longest_run(sequence: str, residues: set[str]) -> float:
        longest = 0
        current = 0
        for aa in sequence:
            if aa in residues:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest / len(sequence)

    @staticmethod
    def _terminal_fraction(
        sequence: str,
        residues: set[str],
        terminal: str,
        width: int,
    ) -> float:
        terminal_seq = sequence[:width] if terminal == "n" else sequence[-width:]
        return sum(aa in residues for aa in terminal_seq) / len(terminal_seq)

    def _reduced_kmer_composition(self, sequences: list[str]) -> pd.DataFrame:
        feature_names = self._reduced_kmer_feature_names()
        rows = []
        for seq in sequences:
            reduced = [self.REDUCED_ALPHABET[aa] for aa in seq]
            row = dict.fromkeys(feature_names, 0.0)
            for k in (2, 3):
                denominator = len(reduced) - k + 1
                if denominator <= 0:
                    continue
                for i in range(denominator):
                    key = "red_kmer_" + "_".join(reduced[i : i + k])
                    row[key] += 1.0 / denominator
            rows.append(row)
        return pd.DataFrame(rows, columns=feature_names)

    @classmethod
    def _reduced_kmer_feature_names(cls) -> list[str]:
        names = []
        for first in cls.REDUCED_SYMBOLS:
            for second in cls.REDUCED_SYMBOLS:
                names.append(f"red_kmer_{first}_{second}")
        for first in cls.REDUCED_SYMBOLS:
            for second in cls.REDUCED_SYMBOLS:
                for third in cls.REDUCED_SYMBOLS:
                    names.append(f"red_kmer_{first}_{second}_{third}")
        return names

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
