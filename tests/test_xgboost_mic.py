from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.mic_runner import train_and_evaluate_mic_baseline
from src.models.registry import MIC_EXPERIMENT_NAMES, get_mic_experiment_spec
from src.models.xgboost_mic import (
    XGBoostMicRegressor,
    aggregate_duplicate_measurements,
    build_regularized_esm2_model,
    build_xgboost_esm2_context_features,
    build_xgboost_amp_core_features,
    build_xgboost_basic_sequence_features,
    build_xgboost_interaction_features,
    build_xgboost_motif_sequence_features,
    build_xgboost_sequence_only_features,
    build_xgboost_taxonomy_gram_features,
    build_xgboost_features,
    load_xgboost_mic_data,
    pca_reduce_esm2_features,
    select_informative_feature_columns,
)
from src.features.plm import save_embedding_cache


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


def test_xgboost_esm2_context_is_registered():
    assert "xgboost_mic_esm2_context" in MIC_EXPERIMENT_NAMES
    assert get_mic_experiment_spec("xgboost_mic_esm2_context").name == (
        "xgboost_mic_esm2_context"
    )


def test_xgboost_esm2_context_selected_is_registered():
    assert "xgboost_mic_esm2_context_selected" in MIC_EXPERIMENT_NAMES
    spec = get_mic_experiment_spec("xgboost_mic_esm2_context_selected")

    assert spec.name == "xgboost_mic_esm2_context_selected"
    assert spec.run_config["feature_transform"] == (
        "train_only_standard_scaler_pca_on_esm2"
    )


def test_xgboost_esm2_context_regularized_is_registered():
    assert "xgboost_mic_esm2_context_regularized" in MIC_EXPERIMENT_NAMES
    spec = get_mic_experiment_spec("xgboost_mic_esm2_context_regularized")

    assert spec.name == "xgboost_mic_esm2_context_regularized"
    assert spec.run_config["regularization_profile"] == "strong_dense_embedding"


def test_regularized_esm2_xgboost_model_uses_stronger_regularization():
    model = build_regularized_esm2_model(random_state=7)

    params = model._model.get_params()
    assert params["max_depth"] == 2
    assert params["min_child_weight"] == 20.0
    assert params["subsample"] == 0.65
    assert params["colsample_bytree"] == 0.35
    assert params["reg_alpha"] == 1.0
    assert params["reg_lambda"] == 25.0
    assert params["learning_rate"] == 0.01
    assert params["early_stopping_rounds"] == 100


def test_load_xgboost_mic_data_filters_invalid_rows(tmp_path):
    path = tmp_path / "taxonomy.csv"
    _taxonomy_frame().to_csv(path, index=False)

    cleaned = load_xgboost_mic_data(path)

    assert cleaned["sequence"].tolist() == ["ACDEFGHIK", "CCCCCCCC"]
    assert np.allclose(cleaned["log_mic"], [1.0, 2.0])


def test_aggregate_duplicate_measurements_uses_median_log_mic():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", "AAAA", "CCCC"],
            "target_activity_name": ["Bacillus", "Bacillus", "Escherichia"],
            "activity": [10.0, 100.0, 1000.0],
            "log_mic": [1.0, 2.0, 3.0],
            "gram_status": ["gram_positive", "gram_positive", "gram_negative"],
        }
    )

    collapsed = aggregate_duplicate_measurements(df)

    assert collapsed["sequence"].tolist() == ["AAAA", "CCCC"]
    assert np.allclose(collapsed["log_mic"], [1.5, 3.0])
    assert np.allclose(collapsed["activity"], [10**1.5, 1000.0])


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


def test_xgboost_sequence_feature_variants_use_smaller_descriptor_sets(tmp_path):
    path = tmp_path / "taxonomy.csv"
    _taxonomy_frame().to_csv(path, index=False)
    cleaned = load_xgboost_mic_data(path)

    basic = build_xgboost_basic_sequence_features(cleaned)
    amp_core = build_xgboost_amp_core_features(cleaned)

    assert "modlamp_charge" in basic.columns
    assert "aa_frac_A" not in basic.columns
    assert "eisenberg_moment" not in basic.columns
    assert "aa_frac_K" in amp_core.columns
    assert "eisenberg_moment" in amp_core.columns
    assert "pepcats_available" not in amp_core.columns
    assert "Genus_Escherichia" in basic.columns
    assert "gram_gram_negative" in amp_core.columns


def test_xgboost_ablation_and_interaction_feature_builders(tmp_path):
    path = tmp_path / "taxonomy.csv"
    _taxonomy_frame().to_csv(path, index=False)
    cleaned = load_xgboost_mic_data(path)

    sequence_only = build_xgboost_sequence_only_features(cleaned)
    taxonomy_gram = build_xgboost_taxonomy_gram_features(cleaned)
    interactions = build_xgboost_interaction_features(cleaned)

    assert "modlamp_charge" in sequence_only.columns
    assert "Genus_Escherichia" not in sequence_only.columns
    assert "Genus_Escherichia" in taxonomy_gram.columns
    assert "modlamp_charge" not in taxonomy_gram.columns
    assert "modlamp_charge_x_gram_gram_negative" in interactions.columns
    assert "local_max_positive_frac_w5_x_Phylum_Bacillota" in interactions.columns
    assert "Genus_Escherichia" in interactions.columns
    assert np.isfinite(interactions.to_numpy()).all()


def test_xgboost_esm2_context_features_load_cache_in_row_order(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "esm2_cache.npz"
    save_embedding_cache(
        path,
        ["CCCCCCCC", "ACDEFGHIK"],
        np.array([[2.0, 2.5], [1.0, 1.5]], dtype=np.float32),
        model_name="facebook/esm2_t12_35M_UR50D",
    )
    monkeypatch.setattr("src.models.xgboost_mic.DEFAULT_MIC_EMBEDDING_PATH", path)

    cleaned = load_xgboost_mic_data_from_frame(_taxonomy_frame())

    features = build_xgboost_esm2_context_features(cleaned)

    assert features[["esm2_0", "esm2_1"]].to_numpy().tolist() == [
        [1.0, 1.5],
        [2.0, 2.5],
    ]
    assert "Genus_Escherichia" in features.columns
    assert "gram_gram_negative" in features.columns
    assert "gram_gram_positive" in features.columns
    assert np.isfinite(features.to_numpy()).all()


def test_xgboost_esm2_context_features_require_complete_cache(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "esm2_cache.npz"
    save_embedding_cache(
        path,
        ["ACDEFGHIK"],
        np.array([[1.0, 1.5]], dtype=np.float32),
        model_name="facebook/esm2_t12_35M_UR50D",
    )
    monkeypatch.setattr("src.models.xgboost_mic.DEFAULT_MIC_EMBEDDING_PATH", path)

    cleaned = load_xgboost_mic_data_from_frame(_taxonomy_frame())

    with pytest.raises(ValueError, match="missing from PLM cache"):
        build_xgboost_esm2_context_features(cleaned)


def test_pca_reduce_esm2_features_fits_train_and_preserves_context():
    X_train = pd.DataFrame(
        {
            "esm2_0": [1.0, 2.0, 3.0, 4.0],
            "esm2_1": [1.0, 1.5, 3.0, 4.5],
            "esm2_2": [4.0, 3.0, 2.0, 1.0],
            "Genus_Bacillus": [1.0, 1.0, 0.0, 0.0],
            "gram_gram_positive": [1.0, 1.0, 0.0, 0.0],
        }
    )
    X_val = pd.DataFrame(
        {
            "esm2_0": [2.0, 5.0],
            "esm2_1": [2.5, 5.5],
            "esm2_2": [3.0, 0.5],
            "Genus_Bacillus": [0.0, 1.0],
            "gram_gram_positive": [0.0, 1.0],
        }
    )

    transformed_train, transformed_splits, metadata = pca_reduce_esm2_features(
        X_train,
        {"val": X_val},
        np.array([1.0, 2.0, 3.0, 4.0]),
        n_components=2,
    )

    assert transformed_train.columns.tolist() == [
        "esm2_pca_0",
        "esm2_pca_1",
        "Genus_Bacillus",
        "gram_gram_positive",
    ]
    assert transformed_splits["val"].columns.tolist() == transformed_train.columns.tolist()
    assert metadata["esm2_original_dim"] == 3
    assert metadata["esm2_pca_components"] == 2
    assert metadata["passthrough_feature_columns"] == [
        "Genus_Bacillus",
        "gram_gram_positive",
    ]
    assert np.isfinite(transformed_train.to_numpy()).all()
    assert np.isfinite(transformed_splits["val"].to_numpy()).all()


def test_xgboost_motif_sequence_features_include_reduced_kmers(tmp_path):
    path = tmp_path / "taxonomy.csv"
    _taxonomy_frame().to_csv(path, index=False)
    cleaned = load_xgboost_mic_data(path)

    features = build_xgboost_motif_sequence_features(cleaned)

    assert "modlamp_charge" in features.columns
    assert "red_kmer_pos_pos" in features.columns
    assert "red_kmer_hyd_hyd_hyd" in features.columns
    assert "Genus_Escherichia" in features.columns
    assert "gram_gram_negative" in features.columns
    assert np.isfinite(features.to_numpy()).all()


def load_xgboost_mic_data_from_frame(df: pd.DataFrame) -> pd.DataFrame:
    from src.models.xgboost_mic import aggregate_duplicate_measurements
    from src.models.mic_baseline import GRAM_CLASSES, NONSTANDARD_PATTERN
    from src.models.taxonomy_mic_baseline import taxonomy_feature_columns

    cleaned = df.copy()
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.upper().str.strip()
    cleaned = cleaned[~cleaned["sequence"].str.contains(NONSTANDARD_PATTERN)]
    cleaned = cleaned[cleaned["target_is_bacteria"].astype(int) == 1]
    cleaned = cleaned[cleaned["gram_status"].isin(GRAM_CLASSES)]
    cleaned["log_mic"] = np.log10(cleaned["activity"])
    for column in taxonomy_feature_columns(cleaned):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0.0)
    return aggregate_duplicate_measurements(cleaned).reset_index(drop=True)


def test_select_informative_feature_columns_drops_constant_and_correlated_columns():
    X = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
            "signal": [0.0, 1.0, 2.0, 3.0, 4.0],
            "signal_copy": [0.0, 1.0, 2.0, 3.0, 4.0],
            "other": [4.0, 1.0, 3.0, 0.0, 2.0],
            "Genus_Bacillus": [1.0, 1.0, 0.0, 0.0, 1.0],
            "gram_gram_positive": [1.0, 1.0, 0.0, 0.0, 1.0],
        }
    )
    y = np.array([0.0, 0.9, 2.1, 2.8, 4.2])

    columns = select_informative_feature_columns(X, y, top_k=3)

    assert "constant" not in columns
    assert not {"signal", "signal_copy"}.issubset(columns)
    assert "Genus_Bacillus" in columns
    assert "gram_gram_positive" in columns
    assert len([column for column in columns if column in {"signal", "other"}]) <= 2


def test_xgboost_regressor_fit_without_validation_preserves_base_interface():
    model = XGBoostMicRegressor(n_estimators=2, early_stopping_rounds=1)

    model.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))

    metadata = model.artifact_metadata(["feature"])
    assert metadata["xgboost_best_iteration"] is None
    assert metadata["xgboost_evals_result"] == {}


def test_xgboost_train_and_evaluate_writes_outputs(tmp_path):
    pytest.importorskip("xgboost")
    train_path = tmp_path / "train.csv"
    output_dir = tmp_path / "results"

    train_df = pd.concat([_taxonomy_frame().iloc[[0, 3]]] * 20, ignore_index=True)
    train_df["sequence"] = [f"ACDEFGHIK{'A' * (i % 8)}" for i in range(len(train_df))]
    train_df["activity"] = np.linspace(1, 40, len(train_df))
    train_df.to_csv(train_path, index=False)

    spec = replace(
        get_mic_experiment_spec("xgboost_mic"),
        build_model=lambda random_state: XGBoostMicRegressor(
            random_state=random_state,
            n_estimators=40,
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
    assert (output_dir / "models" / "xgboost_mic_model.joblib").exists()
    assert (output_dir / "tables" / "xgboost_mic_metrics.csv").exists()
    assert (output_dir / "tables" / "xgboost_mic_predictions.csv").exists()
    artifact = joblib.load(output_dir / "models" / "xgboost_mic_model.joblib")
    assert artifact["xgboost_best_iteration"] is not None
    assert "validation_1" in artifact["xgboost_evals_result"]
    assert "modlamp_charge" in artifact["xgboost_feature_importance_gain"]
