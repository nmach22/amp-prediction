# Prompt Recipes

Use these templates to make future Codex requests concrete.

## Add A MIC Variant

```text
Use $amp-prediction-workflows in this repo. Add a named MIC regression variant called <name> that uses <feature set/model>. Register it in the MIC registry, save metrics/predictions/model artifacts through the shared runner, include artifact metadata for feature columns, and add focused pytest coverage. Use <input csv> for the expected run command and keep W&B optional with --disable-wandb for smoke tests.
```

## Add A Classification Config

```text
Use $amp-prediction-workflows in this repo. Create a YAML classification experiment for <feature encoder> plus <model>. Follow the existing experiments style, include reproducible seed and W&B tags, and update code/tests only if the encoder or model is not already supported. The expected command should be python run_experiment.py --config <config> --disable-wandb.
```

## Review A Model Change

```text
Use $amp-prediction-workflows to review the current changes. Focus on leakage, split correctness, target handling, feature column stability, W&B/artifact reproducibility, and missing tests. Lead with concrete findings and file/line references.
```

## Improve Data Cleaning

```text
Use $amp-prediction-workflows in this repo. Improve the DBAASP/MIC cleaning path for <specific issue>. Preserve current target policy unless explicitly changed, add tests for accepted/rejected rows, and surface any unresolved biological or unit-conversion decisions instead of hard-coding them.
```

## Explain Results For Thesis

```text
Use $amp-prediction-workflows. Read the metrics and prediction tables for <experiment/model>. Summarize the result in thesis-style language: model setup, target, feature set, main metrics, comparison to baseline, limitations, and next experiment to run. Do not invent metric values.
```

## Compare Descriptor Sets

```text
Use $amp-prediction-workflows in this repo. Compare the existing XGBoost MIC descriptor variants: full modlamp, basic sequence, AMP core, and selected sequence. Read the saved metrics tables, report validation MAE/RMSE/R2 and within-fold rates, and recommend the next ablation based only on observed results.
```
