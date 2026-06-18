---
name: amp-prediction-workflows
description: Project workflow guidance for the antimicrobial peptide prediction repository. Use when working on AMP classification, DBAASP data processing, MIC regression, taxonomy or Gram feature engineering, sequence descriptors, experiment YAML files, W&B logging, model artifacts, pytest coverage, or thesis-oriented results summaries for the amp-prediction project.
---

# AMP Prediction Workflows

Use this skill to work efficiently inside the `amp-prediction` repository without rediscovering its experiment structure each time.

## First Pass

Before changing code, inspect the task-relevant parts of the repo:

1. Read `AGENTS.md` and `README.md` for repository rules and known research notes.
2. Check `run_experiment.py` to choose the correct runner path.
3. For MIC model work, inspect `src/models/registry.py`, `src/models/mic_runner.py`, and the model-specific module.
4. For feature work, inspect the relevant `src/features/` module and existing feature tests.
5. For behavior changes, inspect matching tests under `tests/` before editing.

Prefer existing runner, registry, artifact, and testing patterns over new abstractions.

## Workflow Routing

- Use YAML-driven classification when the prompt mentions binary AMP activity, `experiments/*.yml`, `physicochemical`, `word2vec`, `plm`, `SklearnModel`, ROC/AUC, confusion matrices, or fixed train/validation splits.
- Use named MIC regression when the prompt mentions MIC, `log_mic`, Gram status, taxonomy, XGBoost, sequence descriptors, regression metrics, `pred_mic`, or `--model`.
- Use data pipeline guidance when the prompt mentions DBAASP exports, sequence cleaning, duplicate rows, target species, taxonomy enrichment, Gram classification, or unit conversion.
- Use results-writing guidance only after reading the actual metrics, tables, figures, or notebook outputs being summarized.

Load these references as needed:

- `references/project-map.md` for repo layout, canonical commands, and output paths.
- `references/experiment-workflows.md` for adding or modifying classification and MIC experiments.
- `references/data-and-target-policy.md` for sequence, target, duplicate, split, and unresolved research policies.
- `references/prompt-recipes.md` for prompt templates that make future Codex requests more specific.

## Guardrails

- Preserve leakage controls: sequence-grouped MIC splits, train-only feature selection, and no validation/test fitting unless the existing runner explicitly supports validation for early stopping.
- Keep feature columns stable and save enough artifact metadata for inference and interpretation.
- Do not hard-code unresolved research decisions. Surface them instead: MIC unit conversion, `activity_measure` versus concentration, and whether repeated sequences across target species should be collapsed for classification.
- Keep generated raw data, cached embeddings, W&B runs, model artifacts, and large exports out of commits unless the user explicitly asks.

## Verification

Run focused tests for the edited behavior. Use full `pytest` when changes touch shared runners, registries, feature construction, cleaning policy, or metrics.

Common checks:

```bash
pytest tests/test_run_experiment.py
pytest tests/test_sequence_descriptors.py
pytest tests/test_xgboost_mic.py
pytest
```
