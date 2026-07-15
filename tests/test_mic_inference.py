import numpy as np
import pandas as pd
import pytest

from src.models import mic_inference


class RecordingModel:
    def __init__(self):
        self.columns = None

    def predict(self, X):
        self.columns = X.columns.tolist()
        return X["feature_a"].to_numpy() + X["feature_b"].to_numpy()


def test_predict_mic_dataframe_aligns_features_and_adds_mic(monkeypatch):
    model = RecordingModel()
    bundle = {
        "model": model,
        "feature_columns": ["feature_a", "feature_b"],
        "feature_builder": "mlp_physchem",
    }

    monkeypatch.setitem(
        mic_inference.FEATURE_BUILDERS,
        "mlp_physchem",
        lambda df: pd.DataFrame(
            {
                "feature_b": [0.5, 1.5],
                "feature_a": [1.0, 2.0],
                "unused_feature": [99.0, 99.0],
            }
        ),
    )

    predictions = mic_inference.predict_mic_dataframe(
        bundle,
        pd.DataFrame({"sequence": ["AAA", "CCC"]}),
    )

    assert model.columns == ["feature_a", "feature_b"]
    np.testing.assert_allclose(predictions["pred_log_mic"], [1.5, 3.5])
    np.testing.assert_allclose(predictions["pred_mic"], np.power(10.0, [1.5, 3.5]))


def test_predict_mic_dataframe_requires_sequence_column():
    bundle = {
        "model": RecordingModel(),
        "feature_columns": ["feature_a"],
        "feature_builder": "mlp_physchem",
    }

    with pytest.raises(ValueError, match="missing columns"):
        mic_inference.predict_mic_dataframe(bundle, pd.DataFrame({"not_sequence": []}))
