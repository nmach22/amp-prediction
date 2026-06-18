# Experiment Workflows

## Classification Experiments

Use the YAML path when adding or modifying binary AMP activity experiments.

Current shape:

- Configs live in `experiments/`.
- `run_experiment.py --config <path>` loads fixed train/validation splits through `src.data.load_split`.
- `build_features()` supports `onehot`, `physicochemical`, `word2vec`, and `plm`.
- `SklearnModel` wraps classical estimators.
- Metrics come from `src/evaluation/metrics.py`: accuracy, F1, MCC, ROC AUC, PR AUC, sensitivity, specificity, precision.
- Outputs include metrics CSVs, prediction CSVs, model artifacts, ROC curves, confusion matrices, and optional W&B logs.

When adding a classification experiment:

1. Add or reuse a feature encoder in `src/features/`.
2. Add model support only if `SklearnModel` does not already provide it.
3. Create an `experiments/*.yml` config with `experiment_name`, `wandb_project`, `tags`, `features`, `model`, and `seed`.
4. Add tests for config parsing or feature behavior when behavior changes.
5. Smoke test with `python run_experiment.py --config <config> --disable-wandb` when data and dependencies are available.

## MIC Regression Experiments

Use the named model path when the task predicts MIC or `log_mic`.

Current shape:

- Names are listed in `MIC_EXPERIMENT_NAMES`.
- Specs are built in `mic_experiment_specs()`.
- The shared runner writes validation predictions, metrics, and model artifacts.
- `mic_baseline` uses sequence composition plus Gram status with Random Forest.
- `taxonomy_mic_baseline` uses taxonomy features.
- XGBoost variants combine sequence descriptors, taxonomy, and Gram status:
  - `xgboost_mic`: full modlamp descriptors.
  - `xgboost_mic_basic_seq`: compact basic descriptors.
  - `xgboost_mic_amp_core`: AMP-focused descriptors.
  - `xgboost_mic_selected_seq`: train-only feature selection from full descriptors.

When adding a MIC variant:

1. Implement feature construction or model behavior in the relevant `src/models/` or `src/features/` module.
2. Register the variant in `MIC_EXPERIMENT_NAMES` and `mic_experiment_specs()`.
3. Include `run_config` metadata that records target, model, feature set, duplicate policy, and key training settings.
4. Include artifact metadata for feature columns, descriptor set, taxonomy columns, selected columns, and model diagnostics where relevant.
5. Add tests for registration, data filtering, feature columns, runner outputs, and artifact metadata.
6. Smoke test with `python run_experiment.py --model <name> --input <csv> --disable-wandb` when data and dependencies are available.

## Review Checklist

- Does the change preserve sequence-grouped splitting for MIC?
- Is feature selection fit on training data only?
- Are feature columns stable between train and validation?
- Are predictions and metrics saved under the established `results/` layout?
- Is W&B config logging enough resolved model parameters to reproduce the run?
- Are unresolved biology or measurement decisions surfaced instead of silently hard-coded?
