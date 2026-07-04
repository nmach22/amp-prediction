from pathlib import Path

import pytest

from run_experiment import build_wandb_run_config, parse_args
from src.models import SklearnModel
from src.models.registry import get_mic_experiment_spec, mic_experiment_specs


def test_build_wandb_run_config_logs_model_configuration():
    cfg = {
        "experiment_name": "rf_physicochemical",
        "tags": {"approach": "classical-ml"},
        "features": {"name": "physicochemical", "params": {}},
        "model": {
            "name": "random_forest",
            "params": {
                "n_estimators": 10,
                "max_depth": None,
                "random_state": 42,
            },
        },
    }
    model = SklearnModel(
        name=cfg["model"]["name"],
        params=cfg["model"]["params"],
    )

    run_config = build_wandb_run_config(
        cfg=cfg,
        config_file="experiments/rf_physicochemical.yml",
        output_dir=Path("results"),
        seed=42,
        model=model,
    )

    assert run_config["model_config"] == cfg["model"]
    assert run_config["model_n_estimators"] == 10
    assert run_config["resolved_model_params"]["n_estimators"] == 10
    assert run_config["resolved_model_n_estimators"] == 10
    assert "criterion" in run_config["resolved_model_params"]


def test_mic_registry_lists_baseline_models():
    specs = mic_experiment_specs()

    assert {"mic_baseline", "taxonomy_mic_baseline"}.issubset(specs)
    assert {
        "xgboost_mic",
        "xgboost_mic_basic_seq",
        "xgboost_mic_amp_core",
        "xgboost_mic_selected_seq",
        "xgboost_mic_motif_seq",
        "xgboost_mic_sequence_only",
        "xgboost_mic_taxonomy_gram",
        "xgboost_mic_esm2_context",
        "xgboost_mic_esm2_context_selected",
        "xgboost_mic_esm2_context_regularized",
        "xgboost_mic_interactions",
        "catboost_mic_physchem",
        "catboost_mic_tuned",
        "mlp_mic_physchem",
        "mlp_mic_physchem_regularized",
        "mlp_mic_physchem_mild_regularized",
    }.issubset(specs)
    assert get_mic_experiment_spec("mic_baseline").name == "mic_baseline"


def test_parse_args_accepts_named_mic_model(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["run_experiment.py", "--model", "mic_baseline", "--disable-wandb"],
    )

    args = parse_args()

    assert args.model == "mic_baseline"
    assert args.config is None
    assert args.input == "data/processed/splits/train.csv"


def test_parse_args_requires_one_runner_mode(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_experiment.py"])

    with pytest.raises(SystemExit):
        parse_args()
