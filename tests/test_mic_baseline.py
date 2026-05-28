import numpy as np
import pandas as pd

from src.models.mic_baseline import (
    MicBaselineRegressor,
    build_features,
    build_model,
    clean_mic_data,
    encode_sequences,
    evaluate_predictions,
    split_by_sequence,
    split_train_val_by_sequence,
    train_and_evaluate,
)
from src.models import BaseModel


def test_encode_sequences_handles_variable_lengths():
    features = encode_sequences(["ACD", "AAAAAA"])

    assert features.shape == (2, 26)
    assert features.loc[0, "sequence_length"] == 3
    assert features.loc[1, "sequence_length"] == 6
    assert np.isclose(features.loc[0, "aa_frac_A"], 1 / 3)
    assert np.isclose(features.loc[1, "aa_frac_A"], 1.0)


def test_clean_mic_data_filters_invalid_rows_and_adds_log_target():
    raw = pd.DataFrame(
        {
            "sequence": [" acd ", "ACX", "AAAA", "CCCC", "GGGG"],
            "gram_status": [
                "gram_positive",
                "gram_positive",
                "gram_negative",
                "unknown",
                "gram_negative",
            ],
            "activity": [10.0, 12.0, 0.0, 5.0, "100"],
        }
    )

    cleaned = clean_mic_data(raw)

    assert cleaned["sequence"].tolist() == ["ACD", "GGGG"]
    assert cleaned["gram_status"].tolist() == ["gram_positive", "gram_negative"]
    assert np.allclose(cleaned["log_mic"], [1.0, 2.0])


def test_split_by_sequence_prevents_overlap_between_splits():
    df = pd.DataFrame(
        {
            "sequence": [f"SEQ{i}" for i in range(30) for _ in range(2)],
            "gram_status": ["gram_positive", "gram_negative"] * 30,
            "activity": np.linspace(1, 60, 60),
            "log_mic": np.log10(np.linspace(1, 60, 60)),
        }
    )

    splits = split_by_sequence(df, random_state=7)
    train_sequences = set(splits.train["sequence"])
    val_sequences = set(splits.val["sequence"])
    test_sequences = set(splits.test["sequence"])

    assert train_sequences.isdisjoint(val_sequences)
    assert train_sequences.isdisjoint(test_sequences)
    assert val_sequences.isdisjoint(test_sequences)


def test_split_train_val_by_sequence_has_no_test_rows():
    df = pd.DataFrame(
        {
            "sequence": [f"SEQ{i}" for i in range(30) for _ in range(2)],
            "gram_status": ["gram_positive", "gram_negative"] * 30,
            "activity": np.linspace(1, 60, 60),
            "log_mic": np.log10(np.linspace(1, 60, 60)),
        }
    )

    splits = split_train_val_by_sequence(df, random_state=7)

    assert len(splits.train) > 0
    assert len(splits.val) > 0
    assert splits.test.empty
    assert set(splits.train["sequence"]).isdisjoint(set(splits.val["sequence"]))


def test_build_features_adds_stable_gram_columns():
    df = pd.DataFrame(
        {
            "sequence": ["ACDE", "AAAA"],
            "gram_status": ["gram_positive", "gram_positive"],
        }
    )

    features = build_features(df)

    assert "gram_gram_negative" in features.columns
    assert "gram_gram_positive" in features.columns
    assert features["gram_gram_negative"].sum() == 0.0
    assert features["gram_gram_positive"].sum() == 2.0


def test_evaluate_predictions_reports_overall_and_per_gram_metrics():
    df = pd.DataFrame(
        {
            "gram_status": [
                "gram_positive",
                "gram_positive",
                "gram_negative",
                "gram_negative",
            ]
        }
    )
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 2.8, 4.2])

    metrics = evaluate_predictions(df, y_true, y_pred)

    assert {
        "mae",
        "rmse",
        "median_ae",
        "mean_error",
        "r2",
        "pearson",
        "spearman",
        "within_2fold",
        "within_4fold",
        "positive_mae",
        "negative_mae",
    }.issubset(metrics)
    assert metrics["mae"] > 0
    assert 0.0 <= metrics["within_2fold"] <= metrics["within_4fold"] <= 1.0


def test_mic_baseline_regressor_implements_base_model_interface():
    model = build_model(random_state=7)

    assert isinstance(model, MicBaselineRegressor)
    assert isinstance(model, BaseModel)


def test_train_and_evaluate_writes_test_metrics_when_test_csv_is_provided(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    output_dir = tmp_path / "results"
    train_df = pd.DataFrame(
        {
            "sequence": [f"ACD{'A' * (i % 5)}K" for i in range(30)],
            "gram_status": ["gram_positive", "gram_negative"] * 15,
            "activity": np.linspace(1, 30, 30),
        }
    )
    test_df = pd.DataFrame(
        {
            "sequence": ["AAAAK", "CCCCK", "DDDDK", "EEEEK"],
            "gram_status": ["gram_positive", "gram_negative"] * 2,
            "activity": [2.0, 4.0, 8.0, 16.0],
        }
    )
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    metrics = train_and_evaluate(
        input_csv=train_path,
        output_dir=output_dir,
        random_state=7,
        test_csv=test_path,
    )

    saved_metrics = pd.read_csv(output_dir / "tables" / "mic_baseline_metrics.csv")
    predictions = pd.read_csv(output_dir / "tables" / "mic_baseline_predictions.csv")
    assert set(metrics) == {"train", "val", "test"}
    assert saved_metrics["split"].tolist() == ["train", "val", "test"]
    assert {"pearson", "spearman", "within_2fold", "within_4fold"}.issubset(
        saved_metrics.columns
    )
    assert "test" in set(predictions["split"])
