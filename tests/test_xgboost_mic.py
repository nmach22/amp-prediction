import numpy as np
import pandas as pd
import pytest

from src.models.mic_runner import train_and_evaluate_mic_baseline
from src.models.registry import MIC_EXPERIMENT_NAMES, get_mic_experiment_spec
from src.models.xgboost_mic import build_xgboost_features, load_xgboost_mic_data


def _taxonomy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence": ["ACDEFGHIK", "ACX", "AAAA", "CCCCCCCC", "DDDDDDDD"],
            "target_activity_name": [
                "Bacillus subtilis",
                "Bacillus subtilis",
                "Candida albicans",
                "Escherichia coli",
                "Bacillus subtilis",
            ],
            "activity": [10.0, 12.0, 5.0, 100.0, 20.0],
            "gram_status": [
                "gram_positive",
                "gram_positive",
                "non_bacteria",
                "gram_negative",
                "non_bacteria",
            ],
            "Phylum": [
                "Bacillota",
                "Bacillota",
                "Unknown",
                "Pseudomonadota",
                "Bacillota",
            ],
            "Class": [
                "Bacilli",
                "Bacilli",
                "Unknown",
                "Gammaproteobacteria",
                "Bacilli",
            ],
            "Order": [
                "Bacillales",
                "Bacillales",
                "Unknown",
                "Enterobacterales",
                "Bacillales",
            ],
            "Family": [
                "Bacillaceae",
                "Bacillaceae",
                "Unknown",
                "Enterobacteriaceae",
                "Bacillaceae",
            ],
            "Genus": ["Bacillus", "Bacillus", "Unknown", "Escherichia", "Bacillus"],
            "Phylum_Bacillota": [1, 1, 0, 0, 1],
            "Phylum_Pseudomonadota": [0, 0, 0, 1, 0],
            "Genus_Bacillus": [1, 1, 0, 0, 1],
            "Genus_Escherichia": [0, 0, 0, 1, 0],
            "Genus_Unknown": [0, 0, 1, 0, 0],
            "target_is_bacteria": [1, 1, 0, 1, 1],
        }
    )


def test_xgboost_mic_is_registered():
    assert "xgboost_mic" in MIC_EXPERIMENT_NAMES
    assert get_mic_experiment_spec("xgboost_mic").name == "xgboost_mic"


def test_load_xgboost_mic_data_filters_invalid_rows(tmp_path):
    path = tmp_path / "taxonomy.csv"
    _taxonomy_frame().to_csv(path, index=False)

    cleaned = load_xgboost_mic_data(path)

    assert cleaned["sequence"].tolist() == ["ACDEFGHIK", "CCCCCCCC"]
    assert np.allclose(cleaned["log_mic"], [1.0, 2.0])


def test_build_xgboost_features_combines_descriptors_taxonomy_and_gram(tmp_path):
    path = tmp_path / "taxonomy.csv"
    _taxonomy_frame().to_csv(path, index=False)
    cleaned = load_xgboost_mic_data(path)

    features = build_xgboost_features(cleaned)

    assert "modlamp_charge" in features.columns
    assert "Phylum_Bacillota" in features.columns
    assert "Genus_Escherichia" in features.columns
    assert "gram_gram_negative" in features.columns
    assert "gram_gram_positive" in features.columns
    assert np.isfinite(features.to_numpy()).all()


def test_xgboost_train_and_evaluate_writes_outputs(tmp_path):
    pytest.importorskip("xgboost")
    train_path = tmp_path / "train.csv"
    output_dir = tmp_path / "results"

    train_df = pd.concat([_taxonomy_frame().iloc[[0, 3]]] * 20, ignore_index=True)
    train_df["sequence"] = [f"ACDEFGHIK{'A' * (i % 8)}" for i in range(len(train_df))]
    train_df["activity"] = np.linspace(1, 40, len(train_df))
    train_df.to_csv(train_path, index=False)

    metrics = train_and_evaluate_mic_baseline(
        spec=get_mic_experiment_spec("xgboost_mic"),
        input_csv=train_path,
        output_dir=output_dir,
        random_state=7,
    )

    assert set(metrics) == {"train", "val"}
    assert (output_dir / "models" / "xgboost_mic_model.joblib").exists()
    assert (output_dir / "tables" / "xgboost_mic_metrics.csv").exists()
    assert (output_dir / "tables" / "xgboost_mic_predictions.csv").exists()
