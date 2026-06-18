import numpy as np
import pytest

from src.features.sequence_descriptors import SequenceDescriptorEncoder


def test_sequence_descriptor_encoder_returns_stable_numeric_columns():
    encoder = SequenceDescriptorEncoder(pepcats_window=7)

    features = encoder.encode(["ACDEFGHIK", "AAAA"])

    assert features.shape == (2, len(encoder.feature_names()))
    assert features.columns.tolist() == encoder.feature_names()
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[0, "pepcats_available"] == 1.0
    assert features.loc[1, "pepcats_available"] == 0.0


def test_sequence_descriptor_encoder_supports_smaller_feature_sets():
    basic = SequenceDescriptorEncoder(feature_set="basic").encode(["ACDEFGHIK"])
    composition = SequenceDescriptorEncoder(feature_set="composition").encode(["ACDEFGHIK"])
    amp_core = SequenceDescriptorEncoder(feature_set="amp_core").encode(["ACDEFGHIK"])
    interaction_core = SequenceDescriptorEncoder(feature_set="interaction_core").encode(
        ["KKKAAAVVV"]
    )
    motif_core = SequenceDescriptorEncoder(feature_set="motif_core").encode(
        ["KKKAAAVVV"]
    )

    assert "modlamp_charge" in basic.columns
    assert "aa_frac_A" not in basic.columns
    assert "aa_frac_A" in composition.columns
    assert "modlamp_charge" not in composition.columns
    assert "eisenberg_moment" in amp_core.columns
    assert "pepcats_available" not in amp_core.columns
    assert "local_max_positive_frac_w5" in interaction_core.columns
    assert "longest_hydrophobic_run" in interaction_core.columns
    assert "n_terminal_positive_frac" in interaction_core.columns
    assert interaction_core.loc[0, "n_terminal_positive_frac"] == 0.6
    assert "red_kmer_pos_pos" in motif_core.columns
    assert "red_kmer_hyd_hyd_hyd" in motif_core.columns
    assert motif_core.loc[0, "red_kmer_pos_pos"] > 0.0
    assert np.isfinite(basic.to_numpy()).all()
    assert np.isfinite(composition.to_numpy()).all()
    assert np.isfinite(amp_core.to_numpy()).all()
    assert np.isfinite(interaction_core.to_numpy()).all()
    assert np.isfinite(motif_core.to_numpy()).all()


def test_sequence_descriptor_encoder_rejects_unknown_feature_set():
    with pytest.raises(ValueError, match="feature_set"):
        SequenceDescriptorEncoder(feature_set="unknown")


def test_sequence_descriptor_encoder_rejects_nonstandard_sequences():
    encoder = SequenceDescriptorEncoder()

    with pytest.raises(ValueError, match="nonstandard"):
        encoder.encode(["ACX"])
