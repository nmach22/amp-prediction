import numpy as np
import pandas as pd

from scripts.analyze_mic_errors import write_error_analysis


def test_write_error_analysis_groups_prediction_errors(tmp_path):
    predictions = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC", "DDDD", "EEEE"],
            "target_activity_name": ["Bacillus", "Bacillus", "Escherichia", "Escherichia"],
            "log_mic": [1.0, 2.0, 1.5, 2.5],
            "pred_log_mic": [1.1, 1.7, 1.8, 2.0],
            "gram_status": [
                "gram_positive",
                "gram_positive",
                "gram_negative",
                "gram_negative",
            ],
            "Phylum": [
                "Bacillota",
                "Bacillota",
                "Pseudomonadota",
                "Pseudomonadota",
            ],
            "Genus": ["Bacillus", "Bacillus", "Escherichia", "Escherichia"],
        }
    )
    source = pd.DataFrame(
        {
            "sequence": ["AAAA", "AAAA", "CCCC", "DDDD", "EEEE", "EEEE", "EEEE"],
            "target_activity_name": [
                "Bacillus",
                "Bacillus",
                "Bacillus",
                "Escherichia",
                "Escherichia",
                "Escherichia",
                "Escherichia",
            ],
        }
    )
    predictions_path = tmp_path / "predictions.csv"
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "analysis.csv"
    predictions.to_csv(predictions_path, index=False)
    source.to_csv(source_path, index=False)

    analysis = write_error_analysis(
        predictions_path=predictions_path,
        source_path=source_path,
        output_path=output_path,
        min_count=1,
    )

    assert output_path.exists()
    overall = analysis[
        (analysis["segment_type"] == "overall") & (analysis["segment"] == "all")
    ].iloc[0]
    assert np.isclose(overall["mae"], 0.3)
    assert {"gram_status", "phylum", "genus", "duplicate_count_bin"}.issubset(
        set(analysis["segment_type"])
    )
