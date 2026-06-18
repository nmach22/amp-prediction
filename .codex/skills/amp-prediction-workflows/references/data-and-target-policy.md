# Data And Target Policy

## Sequence Rules

- Normalize peptide sequences with uppercase and surrounding whitespace removed.
- Standard amino acid alphabet is `ACDEFGHIKLMNPQRSTVWY`.
- Drop or reject nonstandard sequences unless the user explicitly asks to study them.
- Keep sequence feature encoders deterministic: stable shape, stable column order, finite numeric outputs.

## Classification Target

- Binary AMP classification uses `activity` as a 0/1 label.
- Fixed splits are loaded from `data/processed/splits/{train,val,test}.csv`.
- Metrics should include MCC, ROC AUC, PR AUC, sensitivity, and specificity, not only accuracy.

## MIC Regression Target

- MIC regression uses numeric `activity` values with `log_mic = log10(activity)`.
- Save predictions as both `pred_log_mic` and `pred_mic = 10 ** pred_log_mic`.
- Regression metrics include MAE, RMSE, median absolute error, mean error, R2, Pearson, Spearman, and within-2-fold/within-4-fold rates.
- Per-Gram metrics should be retained when Gram groups exist.

## Duplicate And Split Policy

- MIC splits should group by `sequence` so the same peptide does not appear in both train and validation.
- XGBoost MIC currently collapses duplicate `(sequence, target_activity_name)` rows using median `log_mic`.
- Train-only feature selection is required for selected-feature variants.
- Validation data may be passed to XGBoost only for early stopping through the existing `use_validation_fit` path.

## Taxonomy And Gram Policy

- Gram status classes are `gram_positive` and `gram_negative`.
- Taxonomy features are expected to come from processed taxonomy columns and one-hot prefixes.
- Preserve taxonomy and Gram columns during feature selection unless the user requests an ablation.

## Research Decisions To Surface

Do not resolve these silently:

- Whether MIC values should be normalized to one unit before modeling.
- Whether to use `activity_measure`, concentration, or another DBAASP field as the authoritative MIC source.
- Whether repeated sequences across different target species should be collapsed for classification.
- Whether models should be trained per species, per taxonomy group, per Gram class, or as one multi-target dataset.
- Whether lowercase or mixed-case sequences encode meaningful experimental information.
