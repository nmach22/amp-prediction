"""Model registries used by experiment runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.mic_runner import MicExperimentSpec


MIC_EXPERIMENT_NAMES = ("mic_baseline", "taxonomy_mic_baseline")


def mic_experiment_specs() -> dict[str, MicExperimentSpec]:
    """Return available MIC regression baseline specifications."""
    from src.models.mic_runner import MicExperimentSpec
    from src.models.mic_baseline import (
        build_features,
        evaluate_predictions,
        load_mic_data,
    )
    from src.models.taxonomy_mic_baseline import (
        build_taxonomy_features,
        evaluate_taxonomy_predictions,
        load_taxonomy_mic_data,
        taxonomy_feature_columns,
    )

    return {
        "mic_baseline": MicExperimentSpec(
            name="mic_baseline",
            default_project="mic-baseline",
            default_run_name="mic_baseline_random_forest",
            load_data=load_mic_data,
            build_features=build_features,
            evaluate_predictions=evaluate_predictions,
            prediction_columns=("sequence", "gram_status", "activity", "log_mic"),
            run_config={
                "model_name": "random_forest_regressor",
                "target": "log10_mic",
                "target_features": "sequence_and_gram_status",
            },
        ),
        "taxonomy_mic_baseline": MicExperimentSpec(
            name="taxonomy_mic_baseline",
            default_project="taxonomy-mic-baseline",
            default_run_name="taxonomy_mic_baseline_random_forest",
            load_data=load_taxonomy_mic_data,
            build_features=build_taxonomy_features,
            evaluate_predictions=evaluate_taxonomy_predictions,
            prediction_columns=(
                "sequence",
                "target_activity_name",
                "activity",
                "log_mic",
                "Phylum",
                "Class",
                "Order",
                "Family",
                "Genus",
            ),
            artifact_metadata=lambda df: {
                "taxonomy_feature_columns": taxonomy_feature_columns(df)
            },
            run_config={
                "model_name": "random_forest_regressor",
                "target": "log10_mic",
                "target_features": "taxonomy",
            },
        ),
    }


def get_mic_experiment_spec(name: str) -> MicExperimentSpec:
    """Return one MIC baseline spec by name."""
    specs = mic_experiment_specs()
    if name not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(f"Unknown MIC model '{name}'. Available: {available}")
    return specs[name]
