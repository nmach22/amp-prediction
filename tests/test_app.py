"""End-to-end smoke tests for the FastAPI serving app.

Uses a small fake model artifact (not the real trained MLP) so these tests run fast
and don't depend on a retrained joblib file existing on disk. They still exercise the
real ESM2 model via PLMEncoder, the real request/response wiring, and the real
app/microbes.json lookup table.
"""

from __future__ import annotations

import numpy as np
import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import app.main as app_module

ESM2_DIM = 480  # facebook/esm2_t12_35M_UR50D hidden size


class _FakeModel:
    def predict(self, X):
        return np.array([-1.0])  # log10(MIC) = -1 -> MIC = 0.1 ug/mL


@pytest.fixture()
def fake_artifact_path(tmp_path):
    rng = np.random.default_rng(seed=0)
    esm2_feature_columns = [f"esm2_{i}" for i in range(ESM2_DIM)]
    train_esm2 = pd.DataFrame(
        rng.normal(size=(30, ESM2_DIM)), columns=esm2_feature_columns
    )
    scaler = StandardScaler().fit(train_esm2)
    pca = PCA(n_components=8, random_state=42).fit(scaler.transform(train_esm2))

    passthrough_feature_columns = [
        "physchem_modlamp_length",
        "context_gram_status_gram_negative",
        "context_gram_status_gram_positive",
    ]
    bundle = {
        "model": _FakeModel(),
        "feature_columns": [f"esm2_pca_{i}" for i in range(8)]
        + passthrough_feature_columns,
        "esm2_feature_columns": esm2_feature_columns,
        "esm2_pca_scaler": scaler,
        "esm2_pca_model": pca,
        "passthrough_feature_columns": passthrough_feature_columns,
    }
    path = tmp_path / "fake_model.joblib"
    joblib.dump(bundle, path)
    return path


@pytest.fixture()
def client(fake_artifact_path, monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_PATH", fake_artifact_path)
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_list_microbes_returns_the_fixed_six(client):
    response = client.get("/microbes")
    assert response.status_code == 200
    keys = {microbe["key"] for microbe in response.json()}
    assert keys == {
        "e_coli_atcc_25922",
        "s_aureus_atcc_25923",
        "p_aeruginosa_atcc_27853",
        "a_baumannii_atcc_19606",
        "k_pneumoniae_atcc_700603",
        "e_faecalis_atcc_29212",
    }


def test_predict_returns_mic_for_known_amp(client):
    response = client.post(
        "/predict",
        json={
            "sequence": "GIGKFLHSAKKFGKAFVGEIMNS",
            "microbe_key": "e_coli_atcc_25922",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["microbe_key"] == "e_coli_atcc_25922"
    assert data["log10_mic"] == pytest.approx(-1.0)
    assert data["mic_ug_per_ml"] == pytest.approx(0.1)
    assert data["warnings"] == []


def test_predict_rejects_unknown_microbe(client):
    response = client.post(
        "/predict", json={"sequence": "GIGKFLHSAKKFGKAFVGEIMNS", "microbe_key": "not_real"}
    )
    assert response.status_code == 400


def test_predict_rejects_empty_sequence(client):
    response = client.post(
        "/predict", json={"sequence": "", "microbe_key": "e_coli_atcc_25922"}
    )
    assert response.status_code in (400, 422)


def test_predict_rejects_nonstandard_amino_acids(client):
    response = client.post(
        "/predict",
        json={"sequence": "ACDEFGHIKLMNPQRSTVWX", "microbe_key": "e_coli_atcc_25922"},
    )
    assert response.status_code == 400


def test_predict_warns_on_short_sequence(client):
    response = client.post(
        "/predict", json={"sequence": "AC", "microbe_key": "e_coli_atcc_25922"}
    )
    assert response.status_code == 200
    assert any("short" in warning for warning in response.json()["warnings"])
