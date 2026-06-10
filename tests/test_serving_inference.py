from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.serving.inference import (
    LoadedMicModel,
    MicrobeProfile,
    load_microbes,
    predict_mic,
    validate_sequence,
)

E_COLI = MicrobeProfile(
    key="e_coli_atcc_25922",
    display_name="Escherichia coli (ATCC 25922)",
    target_activity_name="Escherichia coli ATCC 25922",
    gram_status="gram_negative",
    phylum="Pseudomonadota",
    class_="Gammaproteobacteria",
    order="Enterobacterales",
    family="Enterobacteriaceae",
    genus="Escherichia",
)


class _FakeModel:
    """Captures the exact feature matrix passed to predict()."""

    def __init__(self):
        self.last_X: pd.DataFrame | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.last_X = X
        return np.array([1.23])


class _FakePlmEncoder:
    """Stands in for PLMEncoder without requiring torch/transformers."""

    def __init__(self, dim: int = 6):
        self.dim = dim

    def encode(self, sequences, batch_size: int = 32) -> np.ndarray:
        rng = np.random.default_rng(seed=0)
        return rng.normal(size=(len(sequences), self.dim)).astype(np.float32)


def _build_loaded_model(unseen_target_column: str) -> LoadedMicModel:
    esm2_feature_columns = [f"esm2_{i}" for i in range(6)]
    rng = np.random.default_rng(seed=1)
    train_esm2 = pd.DataFrame(
        rng.normal(size=(20, len(esm2_feature_columns))), columns=esm2_feature_columns
    )
    scaler = StandardScaler().fit(train_esm2)
    pca = PCA(n_components=3, random_state=42).fit(scaler.transform(train_esm2))

    passthrough_feature_columns = [
        "physchem_modlamp_length",
        "physchem_target_activity_name_Escherichia coli ATCC 25922",
        unseen_target_column,
        "context_gram_status_gram_negative",
        "context_gram_status_gram_positive",
    ]
    feature_columns = [f"esm2_pca_{i}" for i in range(3)] + passthrough_feature_columns

    return LoadedMicModel(
        model=_FakeModel(),
        feature_columns=feature_columns,
        esm2_feature_columns=esm2_feature_columns,
        esm2_pca_scaler=scaler,
        esm2_pca_model=pca,
        passthrough_feature_columns=passthrough_feature_columns,
    )


def test_predict_mic_zero_fills_unseen_onehot_columns_and_sets_own_columns():
    unseen_column = "physchem_target_activity_name_Staphylococcus aureus ATCC 25923"
    loaded = _build_loaded_model(unseen_column)
    fake_model = loaded.model

    result = predict_mic(loaded, _FakePlmEncoder(dim=6), "GIGKFLHSAKKFGKAFVGEIMNS", E_COLI)

    assert result["log10_mic"] == pytest.approx(1.23)
    assert result["mic_ug_per_ml"] == pytest.approx(10.0**1.23)

    captured = fake_model.last_X
    assert list(captured.columns) == loaded.feature_columns
    # The category actually present in this row's one-hot is 1.0 ...
    assert captured["physchem_target_activity_name_Escherichia coli ATCC 25922"].iloc[0] == 1.0
    assert captured["context_gram_status_gram_negative"].iloc[0] == 1.0
    # ... while a category seen in training but absent from this single-row batch is zero-filled.
    assert captured[unseen_column].iloc[0] == 0.0
    assert captured["context_gram_status_gram_positive"].iloc[0] == 0.0


def test_predict_mic_applies_persisted_pca_deterministically():
    loaded = _build_loaded_model("physchem_target_activity_name_Some Other Strain")
    plm_encoder = _FakePlmEncoder(dim=6)

    first = predict_mic(loaded, plm_encoder, "GIGKFLHSAKKFGKAFVGEIMNS", E_COLI)
    second = predict_mic(loaded, plm_encoder, "GIGKFLHSAKKFGKAFVGEIMNS", E_COLI)

    # Same sequence/microbe through the same fitted scaler+PCA -> identical PCA columns.
    assert first == second


@pytest.mark.parametrize(
    "sequence",
    ["", "   ", "ACDEFGHIKLMNPQRSTVWX", "acdefghiklm123"],
)
def test_validate_sequence_rejects_invalid_input(sequence):
    with pytest.raises(ValueError):
        validate_sequence(sequence)


def test_validate_sequence_warns_on_short_and_long_sequences():
    assert any("short" in warning for warning in validate_sequence("AC"))
    assert any("long" in warning for warning in validate_sequence("A" * 80))
    assert validate_sequence("GIGKFLHSAKKFGKAFVGEIMNS") == []


def test_load_microbes_reads_fixed_lookup_table():
    microbes_path = Path(__file__).resolve().parents[1] / "app" / "microbes.json"
    microbes = load_microbes(microbes_path)

    assert len(microbes) == 6
    assert "e_coli_atcc_25922" in microbes
    e_coli = microbes["e_coli_atcc_25922"]
    assert e_coli.target_activity_name == "Escherichia coli ATCC 25922"
    assert e_coli.gram_status == "gram_negative"
    assert e_coli.genus == "Escherichia"
