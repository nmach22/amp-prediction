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
from src.models.mlp_mic import (
    MlpMicRegressor,
    build_esm2_context_regularized_model,
    build_mlp_esm2_context_features,
    build_mlp_features,
    build_mlp_physchem_esm2_context_features,
    build_mild_regularized_model,
    build_physchem_esm2_context_regularized_model,
    build_regularized_model,
    load_mlp_mic_data,
)
from src.features.plm import save_embedding_cache
from src.models.registry import MIC_EXPERIMENT_NAMES, get_mic_experiment_spec
from src.models.xgboost_mic import load_xgboost_mic_data
from src.utils.wandb_logging import group_metric_history


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
    assert "mlp_mic_physchem_regularized" in MIC_EXPERIMENT_NAMES
    assert "mlp_mic_physchem_mild_regularized" in MIC_EXPERIMENT_NAMES
    assert "mlp_mic_esm2_context_regularized" in MIC_EXPERIMENT_NAMES
    assert "mlp_mic_physchem_esm2_context_regularized" in MIC_EXPERIMENT_NAMES
    assert get_mic_experiment_spec("catboost_mic_physchem").use_validation_fit
    assert get_mic_experiment_spec("catboost_mic_tuned").use_validation_fit
    assert get_mic_experiment_spec("mlp_mic_physchem").use_validation_fit
    assert get_mic_experiment_spec("mlp_mic_physchem_regularized").use_validation_fit
    assert get_mic_experiment_spec(
        "mlp_mic_physchem_mild_regularized"
    ).use_validation_fit
    assert get_mic_experiment_spec("mlp_mic_esm2_context_regularized").use_validation_fit
    assert get_mic_experiment_spec(
        "mlp_mic_physchem_esm2_context_regularized"
    ).use_validation_fit


def test_mlp_esm2_context_uses_embedding_cache_compatible_loader():
    spec = get_mic_experiment_spec("mlp_mic_esm2_context_regularized")

    assert spec.load_data is load_xgboost_mic_data


def test_mlp_physchem_esm2_context_uses_embedding_cache_compatible_loader():
    spec = get_mic_experiment_spec("mlp_mic_physchem_esm2_context_regularized")

    assert spec.load_data is load_xgboost_mic_data


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


def test_mlp_esm2_context_features_load_cache_and_context(tmp_path, monkeypatch):
    path = tmp_path / "esm2_cache.npz"
    save_embedding_cache(
        path,
        ["CCCCCCCC", "ACDEFGHIK", "DDDDDDDD"],
        np.array(
            [[2.0, 2.5], [1.0, 1.5], [3.0, 3.5]],
            dtype=np.float32,
        ),
        model_name="facebook/esm2_t12_35M_UR50D",
    )
    monkeypatch.setattr("src.models.mlp_mic.DEFAULT_MIC_EMBEDDING_PATH", path)
    cleaned = load_mlp_mic_data_from_frame(_raw_frame())

    features = build_mlp_esm2_context_features(cleaned)

    assert features[["esm2_0", "esm2_1"]].to_numpy().tolist() == [
        [1.0, 1.5],
        [2.0, 2.5],
        [3.0, 3.5],
    ]
    assert "gram_status_gram_positive" in features.columns
    assert "Genus_Bacillus" in features.columns
    assert "target_activity_name_Bacillus subtilis" not in features.columns
    assert np.isfinite(features.to_numpy()).all()


def test_mlp_physchem_esm2_context_features_combine_feature_families(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "esm2_cache.npz"
    save_embedding_cache(
        path,
        ["CCCCCCCC", "ACDEFGHIK", "DDDDDDDD"],
        np.array(
            [[2.0, 2.5], [1.0, 1.5], [3.0, 3.5]],
            dtype=np.float32,
        ),
        model_name="facebook/esm2_t12_35M_UR50D",
    )
    monkeypatch.setattr("src.models.mlp_mic.DEFAULT_MIC_EMBEDDING_PATH", path)
    cleaned = load_mlp_mic_data_from_frame(_raw_frame())

    features = build_mlp_physchem_esm2_context_features(cleaned)

    assert "physchem_modlamp_charge" in features.columns
    assert "physchem_eng_charge_per_length" in features.columns
    assert "esm2_0" in features.columns
    assert "context_gram_status_gram_positive" in features.columns
    assert "context_Genus_Bacillus" in features.columns
    assert np.isfinite(features.to_numpy()).all()


def load_mlp_mic_data_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    from src.models.catboost_mic import (
        TAXONOMY_RANK_COLUMNS,
        aggregate_duplicate_measurements,
    )
    from src.models.mic_baseline import GRAM_CLASSES, NONSTANDARD_PATTERN

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["sequence", "target_activity_name", "activity"])
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.upper().str.strip()
    cleaned["target_activity_name"] = (
        cleaned["target_activity_name"].astype(str).str.strip()
    )
    cleaned["gram_status"] = cleaned["gram_status"].astype(str).str.strip()
    cleaned["activity"] = pd.to_numeric(cleaned["activity"], errors="coerce")
    cleaned = cleaned.dropna(subset=["activity"])
    cleaned = cleaned[cleaned["activity"] > 0]
    cleaned = cleaned[cleaned["sequence"].str.len() > 0]
    cleaned = cleaned[~cleaned["sequence"].str.contains(NONSTANDARD_PATTERN)]
    cleaned = cleaned[cleaned["gram_status"].isin(GRAM_CLASSES)].copy()
    for column in TAXONOMY_RANK_COLUMNS:
        cleaned[column] = (
            cleaned[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
    cleaned["log_mic"] = np.log10(cleaned["activity"])
    return aggregate_duplicate_measurements(cleaned.reset_index(drop=True))


def test_regularized_mlp_model_uses_intended_hyperparameters():
    pytest.importorskip("torch")

    model = build_regularized_model(random_state=7)

    assert model.random_state == 7
    assert model.hidden_layers == (128, 64, 32)
    assert model.dropout == 0.35
    assert model.weight_decay == 5e-4
    assert model.learning_rate == 5e-4
    assert model.max_epochs == 400
    assert model.patience == 20
    assert model.noise_std == 0.01


def test_mild_regularized_mlp_model_uses_intended_hyperparameters():
    pytest.importorskip("torch")

    model = build_mild_regularized_model(random_state=7)

    assert model.random_state == 7
    assert model.hidden_layers == (192, 96, 48)
    assert model.dropout == 0.25
    assert model.weight_decay == 2e-4
    assert model.learning_rate == 7e-4
    assert model.max_epochs == 400
    assert model.patience == 25
    assert model.noise_std == 0.005


def test_esm2_context_regularized_mlp_model_uses_intended_hyperparameters():
    pytest.importorskip("torch")

    model = build_esm2_context_regularized_model(random_state=7)

    assert model.random_state == 7
    assert model.hidden_layers == (128, 64)
    assert model.dropout == 0.4
    assert model.weight_decay == 1e-3
    assert model.learning_rate == 5e-4
    assert model.max_epochs == 400
    assert model.patience == 25
    assert model.noise_std == 0.01


def test_physchem_esm2_context_regularized_mlp_model_uses_intended_hyperparameters():
    pytest.importorskip("torch")

    model = build_physchem_esm2_context_regularized_model(random_state=7)

    assert model.random_state == 7
    assert model.hidden_layers == (192, 96, 48)
    assert model.dropout == 0.35
    assert model.weight_decay == 7e-4
    assert model.learning_rate == 5e-4
    assert model.max_epochs == 450
    assert model.patience == 30
    assert model.noise_std == 0.01


def test_regularized_mlp_noise_is_train_only(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    train_path = tmp_path / "train.csv"
    _training_frame().to_csv(train_path, index=False)
    cleaned = load_mlp_mic_data(train_path)
    train = cleaned.iloc[:24]
    val = cleaned.iloc[24:]
    X_train = build_mlp_features(train)
    X_val = build_mlp_features(val).reindex(columns=X_train.columns, fill_value=0.0)
    y_train = train["log_mic"].to_numpy()
    y_val = val["log_mic"].to_numpy()
    model = MlpMicRegressor(
        random_state=7,
        hidden_layers=(16, 8),
        max_epochs=2,
        patience=2,
        batch_size=8,
        noise_std=0.05,
    )
    calls = {"count": 0}
    original_add_noise = model._add_training_noise

    def count_noise(X_batch):
        calls["count"] += 1
        return original_add_noise(X_batch)

    monkeypatch.setattr(model, "_add_training_noise", count_noise)

    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    first_pred = model.predict(X_val)
    second_pred = model.predict(X_val)

    assert calls["count"] > 0
    assert np.allclose(first_pred, second_pred)


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
    metrics, metric_history = train_and_evaluate_mic_baseline(
        spec=spec,
        input_csv=train_path,
        output_dir=output_dir,
        random_state=7,
        return_history=True,
    )

    assert set(metrics) == {"train", "val"}
    artifact = joblib.load(output_dir / "models" / "mlp_mic_physchem_model.joblib")
    assert artifact["mlp_best_epoch"] is not None
    assert "numeric_imputation_medians" in artifact
    history = artifact["mlp_training_history"]
    assert history
    expected_history_keys = {
        "epoch",
        "train_loss",
        "train_mae",
        "train_rmse",
        "train_r2",
        "val_loss",
        "val_mae",
        "val_rmse",
        "val_r2",
    }
    assert expected_history_keys.issubset(history[0])
    assert len(metric_history) == len(history) * 2
    for row in metric_history:
        assert {"loss", "mae", "rmse", "r2"}.issubset(row["metrics"])
    grouped_history = group_metric_history(metric_history)
    assert {
        "train/loss",
        "train/mae",
        "train/rmse",
        "train/r2",
        "val/loss",
        "val/mae",
        "val/rmse",
        "val/r2",
    }.issubset(grouped_history[0])
    assert (output_dir / "tables" / "mlp_mic_physchem_predictions.csv").exists()


def test_regularized_mlp_train_and_evaluate_writes_outputs(tmp_path):
    pytest.importorskip("torch")
    train_path = tmp_path / "train.csv"
    output_dir = tmp_path / "results"
    _training_frame().to_csv(train_path, index=False)

    spec = replace(
        get_mic_experiment_spec("mlp_mic_physchem_regularized"),
        build_model=lambda random_state: MlpMicRegressor(
            random_state=random_state,
            hidden_layers=(16, 8),
            dropout=0.35,
            learning_rate=5e-4,
            weight_decay=5e-4,
            max_epochs=3,
            patience=2,
            batch_size=8,
            noise_std=0.01,
        ),
    )
    metrics, metric_history = train_and_evaluate_mic_baseline(
        spec=spec,
        input_csv=train_path,
        output_dir=output_dir,
        random_state=7,
        return_history=True,
    )

    assert set(metrics) == {"train", "val"}
    artifact = joblib.load(
        output_dir / "models" / "mlp_mic_physchem_regularized_model.joblib"
    )
    assert artifact["mlp_hidden_layers"] == [16, 8]
    assert artifact["mlp_dropout"] == 0.35
    assert artifact["mlp_weight_decay"] == 5e-4
    assert artifact["mlp_learning_rate"] == 5e-4
    assert artifact["mlp_noise_std"] == 0.01
    assert artifact["mlp_best_epoch"] is not None
    assert artifact["mlp_train_mae_at_best_epoch"] is not None
    assert artifact["mlp_best_validation_mae"] is not None
    assert artifact["mlp_train_val_mae_gap_at_best_epoch"] is not None
    history = artifact["mlp_training_history"]
    assert history
    for row in metric_history:
        assert {"loss", "mae", "rmse", "r2"}.issubset(row["metrics"])
    assert (
        output_dir / "tables" / "mlp_mic_physchem_regularized_metrics.csv"
    ).exists()
    assert (
        output_dir / "tables" / "mlp_mic_physchem_regularized_predictions.csv"
    ).exists()


def test_mild_regularized_mlp_train_and_evaluate_writes_outputs(tmp_path):
    pytest.importorskip("torch")
    train_path = tmp_path / "train.csv"
    output_dir = tmp_path / "results"
    _training_frame().to_csv(train_path, index=False)

    spec = replace(
        get_mic_experiment_spec("mlp_mic_physchem_mild_regularized"),
        build_model=lambda random_state: MlpMicRegressor(
            random_state=random_state,
            hidden_layers=(16, 8),
            dropout=0.25,
            learning_rate=7e-4,
            weight_decay=2e-4,
            max_epochs=3,
            patience=2,
            batch_size=8,
            noise_std=0.005,
        ),
    )
    metrics, metric_history = train_and_evaluate_mic_baseline(
        spec=spec,
        input_csv=train_path,
        output_dir=output_dir,
        random_state=7,
        return_history=True,
    )

    assert set(metrics) == {"train", "val"}
    artifact = joblib.load(
        output_dir / "models" / "mlp_mic_physchem_mild_regularized_model.joblib"
    )
    assert artifact["mlp_hidden_layers"] == [16, 8]
    assert artifact["mlp_dropout"] == 0.25
    assert artifact["mlp_weight_decay"] == 2e-4
    assert artifact["mlp_learning_rate"] == 7e-4
    assert artifact["mlp_noise_std"] == 0.005
    assert artifact["mlp_train_val_mae_gap_at_best_epoch"] is not None
    for row in metric_history:
        assert {"loss", "mae", "rmse", "r2"}.issubset(row["metrics"])
    assert (
        output_dir / "tables" / "mlp_mic_physchem_mild_regularized_metrics.csv"
    ).exists()
    assert (
        output_dir / "tables" / "mlp_mic_physchem_mild_regularized_predictions.csv"
    ).exists()
