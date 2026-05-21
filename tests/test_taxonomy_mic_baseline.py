import numpy as np
import pandas as pd

from src.models.taxonomy_mic_baseline import (
    build_taxonomy_features,
    load_taxonomy_mic_data,
    taxonomy_feature_columns,
)


def test_taxonomy_feature_columns_detects_rank_features():
    df = pd.DataFrame(
        {
            "Phylum_Bacillota": [1],
            "Class_Bacilli": [1],
            "target_is_bacteria": [1],
            "gram_gram_positive": [1],
        }
    )

    assert taxonomy_feature_columns(df) == [
        "Phylum_Bacillota",
        "Class_Bacilli",
        "target_is_bacteria",
    ]


def test_load_taxonomy_mic_data_filters_invalid_and_non_bacteria(tmp_path):
    path = tmp_path / "taxonomy.csv"
    raw = pd.DataFrame(
        {
            "sequence": [" acd ", "ACX", "AAAA", "CCCC", "GGGG"],
            "target_activity_name": [
                "Bacillus subtilis",
                "Bacillus subtilis",
                "Candida albicans",
                "Bacillus subtilis",
                "Bacillus subtilis",
            ],
            "activity": [10.0, 12.0, 5.0, 0.0, "100"],
            "Phylum": ["Bacillota", "Bacillota", "Unknown", "Bacillota", "Bacillota"],
            "Genus": ["Bacillus", "Bacillus", "Unknown", "Bacillus", "Bacillus"],
            "Phylum_Bacillota": [1, 1, 0, 1, 1],
            "Genus_Bacillus": [1, 1, 0, 1, 1],
            "Genus_Unknown": [0, 0, 1, 0, 0],
            "target_is_bacteria": [1, 1, 0, 1, 1],
        }
    )
    raw.to_csv(path, index=False)

    cleaned = load_taxonomy_mic_data(path)

    assert cleaned["sequence"].tolist() == ["ACD", "GGGG"]
    assert cleaned["target_activity_name"].tolist() == [
        "Bacillus subtilis",
        "Bacillus subtilis",
    ]
    assert np.allclose(cleaned["log_mic"], [1.0, 2.0])


def test_build_taxonomy_features_combines_sequence_and_taxonomy_features():
    df = pd.DataFrame(
        {
            "sequence": ["ACDE", "AAAA"],
            "Phylum_Bacillota": [1, 1],
            "Genus_Bacillus": [1, 1],
            "target_is_bacteria": [1, 1],
        }
    )

    features = build_taxonomy_features(df)

    assert "sequence_length" in features.columns
    assert "aa_frac_A" in features.columns
    assert "Phylum_Bacillota" in features.columns
    assert "Genus_Bacillus" in features.columns
    assert "target_is_bacteria" in features.columns
    assert features.loc[0, "sequence_length"] == 4
