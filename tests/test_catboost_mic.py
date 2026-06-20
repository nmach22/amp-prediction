from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.catboost_mic import (
    CatBoostMicRegressor,
    aggregate_duplicate_measurements,
    build_catboost_features,
    build_tuned_model,
    load_catboost_mic_data,
)
from src.models.mic_runner import train_and_evaluate_mic_baseline
from src.models.mlp_mic import MlpMicRegressor, build_mlp_features, load_mlp_mic_data
from src.models.registry import MIC_EXPERIMENT_NAMES, get_mic_experiment_spec


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence": [
                " ACDEFGHIK ",
                "ACDEFGHIK",
                "ACX",
                "AAAA",
                "CCCCCCCC",
                "DDDDDDDD",
            ],
            "target_activity_name": [
                "Bacillus subtilis",
                "Bacillus subtilis",
                "Bacillus subtilis",
                "Candida albicans",
                "Escherichia coli",
                "Bacillus subtilis",
            ],
            "activity": [10.0, 100.0, 12.0, 5.0, 100.0, 20.0],
            "gram_status": [
                "gram_positive",
                "gram_positive",
                "gram_positive",
                "non_bacteria",
                "gram_negative",
                "gram_positive",
            ],
            "Phylum": [
                "Bacillota",
                None,
                "Bacillota",
                "Unknown",
                "Pseudomonadota",
                "Bacillota",
            ],
            "Class": [
                "Bacilli",
                "Bacilli",
                "Bacilli",
                "Unknown",
                "Gammaproteobacteria",
                "Bacilli",
            ],
            "Order": [
                "Bacillales",
                "Bacillales",
                "Bacillales",
                "Unknown",
                "Enterobacterales",
                "Bacillales",
            ],
            "Family": [
                "Bacillaceae",
                "Bacillaceae",
                "Bacillaceae",
                "Unknown",
                "Enterobacteriaceae",
                "Bacillaceae",
            ],
            "Genus": [
                "Bacillus",
                "Bacillus",
                "Bacillus",
                "Unknown",
                "Escherichia",
                "Bacillus",
            ],
        }
    )


def _training_frame(n_rows: int = 36) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        gram_status = "gram_positive" if i % 2 == 0 else "gram_negative"
        genus = "Bacillus" if gram_status == "gram_positive" else "Escherichia"
        rows.append(
            {
                "sequence": f"ACDEFGHIK{'A' * (i % 7)}{'K' * (i % 3)}",
                "target_activity_name": f"{genus} species {i % 5}",
                "activity": float(i + 1),
                "gram_status": gram_status,
                "Phylum": "Bacillota"
                if gram_status == "gram_positive"
                else "Pseudomonadota",
                "Class": "Bacilli"
                if gram_status == "gram_positive"
                else "Gammaproteobacteria",
                "Order": "Bacillales"
                if gram_status == "gram_positive"
                else "Enterobacterales",
                "Family": "Bacillaceae"
                if gram_status == "gram_positive"
                else "Enterobacteriaceae",
                "Genus": genus,
            }
        )
    return pd.DataFrame(rows)


def test_new_mic_models_are_registered():
    assert "catboost_mic_physchem" in MIC_EXPERIMENT_NAMES
    assert "catboost_mic_tuned" in MIC_EXPERIMENT_NAMES
    assert "mlp_mic_physchem" in MIC_EXPERIMENT_NAMES
    assert get_mic_experiment_spec("catboost_mic_physchem").use_validation_fit
    assert get_mic_experiment_spec("catboost_mic_tuned").use_validation_fit
    assert get_mic_experiment_spec("mlp_mic_physchem").use_validation_fit


def test_tuned_catboost_model_uses_mae_objective():
    pytest.importorskip("catboost")

    model = build_tuned_model(random_state=7)

    params = model._model.get_params()
    assert params["loss_function"] == "MAE"
    assert params["eval_metric"] == "MAE"
    assert params["depth"] == 7
    assert params["early_stopping_rounds"] == 150


def test_load_catboost_mic_data_cleans_nulls_and_duplicates(tmp_path):
    path = tmp_path / "raw.csv"
    _raw_frame().to_csv(path, index=False)

    cleaned = load_catboost_mic_data(path)

    assert cleaned["sequence"].tolist() == ["ACDEFGHIK", "CCCCCCCC", "DDDDDDDD"]
    assert cleaned.loc[0, "Phylum"] == "Bacillota"
    assert np.allclose(cleaned["log_mic"], [1.5, 2.0, np.log10(20.0)])
    assert np.allclose(cleaned["activity"].iloc[0], 10**1.5)


def test_aggregate_duplicate_measurements_preserves_first_metadata():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", "AAAA", "CCCC"],
            "target_activity_name": ["Bacillus", "Bacillus", "Escherichia"],
            "activity": [10.0, 100.0, 1000.0],
            "log_mic": [1.0, 2.0, 3.0],
            "gram_status": ["gram_positive", "gram_positive", "gram_negative"],
            "Phylum": ["Bacillota", "Changed", "Pseudomonadota"],
        }
    )

    collapsed = aggregate_duplicate_measurements(df)

    assert collapsed["sequence"].tolist() == ["AAAA", "CCCC"]
    assert collapsed.loc[0, "Phylum"] == "Bacillota"
    assert np.allclose(collapsed["log_mic"], [1.5, 3.0])


def test_catboost_features_include_engineered_and_categorical_columns(tmp_path):
    path = tmp_path / "raw.csv"
    _raw_frame().to_csv(path, index=False)
    cleaned = load_catboost_mic_data(path)

    features = build_catboost_features(cleaned)

    assert "modlamp_charge" in features.columns
    assert "red_kmer_pos_pos" in features.columns
    assert "eng_charge_per_length" in features.columns
    assert "eng_positive_minus_negative_frac" in features.columns
    assert "gram_status" in features.columns
    assert "Genus" in features.columns
    numeric = features.drop(columns=["target_activity_name", "gram_status", "Phylum", "Class", "Order", "Family", "Genus"])
    assert np.isfinite(numeric.to_numpy()).all()


def test_mlp_features_are_numeric_one_hot_features(tmp_path):
    path = tmp_path / "raw.csv"
    _raw_frame().to_csv(path, index=False)
    cleaned = load_mlp_mic_data(path)

    features = build_mlp_features(cleaned)

    assert "modlamp_charge" in features.columns
    assert "eng_charge_per_length" in features.columns
    assert "gram_status_gram_positive" in features.columns
    assert "Genus_Bacillus" in features.columns
    assert np.isfinite(features.to_numpy()).all()


def test_catboost_train_and_evaluate_writes_outputs(tmp_path):
    pytest.importorskip("catboost")
    train_path = tmp_path / "train.csv"
    output_dir = tmp_path / "results"
    _training_frame().to_csv(train_path, index=False)

    spec = replace(
        get_mic_experiment_spec("catboost_mic_physchem"),
        build_model=lambda random_state: CatBoostMicRegressor(
            random_state=random_state,
            iterations=20,
            early_stopping_rounds=5,
        ),
    )
    metrics = train_and_evaluate_mic_baseline(
        spec=spec,
        input_csv=train_path,
        output_dir=output_dir,
        random_state=7,
    )

    assert set(metrics) == {"train", "val"}
    artifact = joblib.load(output_dir / "models" / "catboost_mic_physchem_model.joblib")
    assert artifact["catboost_best_iteration"] is not None
    assert "catboost_categorical_columns" in artifact
    assert "eng_charge_per_length" in artifact["feature_columns"]
    assert (output_dir / "tables" / "catboost_mic_physchem_predictions.csv").exists()


def test_mlp_train_and_evaluate_writes_outputs(tmp_path):
    pytest.importorskip("torch")
    train_path = tmp_path / "train.csv"
    output_dir = tmp_path / "results"
    _training_frame().to_csv(train_path, index=False)

    spec = replace(
        get_mic_experiment_spec("mlp_mic_physchem"),
        build_model=lambda random_state: MlpMicRegressor(
            random_state=random_state,
            hidden_layers=(16, 8),
            max_epochs=3,
            patience=2,
            batch_size=8,
        ),
    )
    metrics = train_and_evaluate_mic_baseline(
        spec=spec,
        input_csv=train_path,
        output_dir=output_dir,
        random_state=7,
    )

    assert set(metrics) == {"train", "val"}
    artifact = joblib.load(output_dir / "models" / "mlp_mic_physchem_model.joblib")
    assert artifact["mlp_best_epoch"] is not None
    assert "numeric_imputation_medians" in artifact
    assert (output_dir / "tables" / "mlp_mic_physchem_predictions.csv").exists()
