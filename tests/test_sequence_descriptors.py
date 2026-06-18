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


def test_sequence_descriptor_encoder_rejects_nonstandard_sequences():
    encoder = SequenceDescriptorEncoder()

    with pytest.raises(ValueError, match="nonstandard"):
        encoder.encode(["ACX"])
