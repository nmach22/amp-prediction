# Frozen ESM2 MIC Regression Experiments

Date: 2026-07-15

This report summarizes the MIC regression experiments performed after adding frozen protein language model features from `facebook/esm2_t12_35M_UR50D`. The goal was to test whether pretrained peptide embeddings improve MIC prediction when used alone or together with existing physicochemical and biological context features.

## Experimental Setup

The ESM2 model was used as a frozen encoder. Peptide sequences were passed through the pretrained model once, residue representations were mean-pooled, and the resulting sequence-level embeddings were saved in an `.npz` cache under `data/processed/embeddings/`. During model training, embeddings are loaded from this cache rather than recomputed, avoiding repeated transformer inference.

The target variable is log-transformed MIC. Therefore, MAE and RMSE are reported in log10 space. A validation MAE of approximately 0.39 corresponds to an average error of about `10^0.39 = 2.45`-fold in MIC.

The experiments compare:

- XGBoost models using ESM2 embeddings and organism context.
- MLP models using ESM2 embeddings.
- MLP models using the original physicochemical descriptors.
- MLP models combining physicochemical descriptors, ESM2 embeddings, and taxonomy/Gram context.
- PCA-compressed ESM2 variants to reduce embedding dimensionality and noise.

All reported results below are from the saved metric files in `results/tables/`. They should be interpreted as single-seed validation results unless otherwise noted.

## Results

| Model | Feature set | Train MAE | Train RMSE | Train R2 | Val MAE | Val RMSE | Val R2 |
|---|---|---:|---:|---:|---:|---:|---:|
| `xgboost_mic_esm2_context` | ESM2 + context | 0.3545 | 0.4601 | 0.6743 | 0.4899 | 0.6440 | 0.3868 |
| `xgboost_mic_esm2_context_selected` | selected ESM2/context features | 0.3682 | 0.4780 | 0.6486 | 0.4986 | 0.6504 | 0.3747 |
| `xgboost_mic_esm2_context_regularized` | regularized ESM2 + context | 0.4907 | 0.6216 | 0.4056 | 0.5321 | 0.6831 | 0.3101 |
| `mlp_mic_esm2_context_regularized` | ESM2 + context | 0.3577 | 0.4852 | 0.6379 | 0.4650 | 0.6272 | 0.4185 |
| `mlp_mic_physchem_mild_regularized` | physicochemical descriptors | 0.1792 | 0.2509 | 0.9039 | 0.4028 | 0.5576 | 0.5238 |
| `mlp_mic_physchem_esm2_context_regularized` | physicochemical + ESM2 + context | 0.1849 | 0.2600 | 0.8966 | 0.3867 | 0.5442 | 0.5519 |
| `mlp_mic_physchem_esm2_pca_context_regularized` | physicochemical + PCA-ESM2 + context | 0.1546 | 0.2198 | 0.9257 | 0.3910 | 0.5570 | 0.5412 |
| `mlp_mic_physchem_esm2_pca_context_strong_regularized` | stronger-regularized physicochemical + PCA-ESM2 + context | 0.2180 | 0.3100 | 0.8522 | 0.4023 | 0.5654 | 0.5274 |

## Main Comparisons

### Frozen ESM2 alone was not enough

The ESM2-only/context models did not outperform the physicochemical baseline. The best ESM2-only validation result was obtained by the MLP:

| Model | Val MAE | Val RMSE | Val R2 |
|---|---:|---:|---:|
| `mlp_mic_esm2_context_regularized` | 0.4650 | 0.6272 | 0.4185 |
| `xgboost_mic_esm2_context` | 0.4899 | 0.6440 | 0.3868 |
| `mlp_mic_physchem_mild_regularized` | 0.4028 | 0.5576 | 0.5238 |

This suggests that the frozen ESM2 embeddings contain useful peptide information, but in this dataset they do not replace the hand-crafted physicochemical descriptors.

### MLP handled ESM2 features better than XGBoost

For the ESM2 + context setup, the MLP improved over XGBoost:

- Validation MAE improved from 0.4899 to 0.4650.
- Validation RMSE improved from 0.6440 to 0.6272.
- Validation R2 improved from 0.3868 to 0.4185.

This is plausible because dense ESM2 embeddings are continuous neural features, and a neural downstream model may exploit them more naturally than a tree model.

### Stronger XGBoost regularization caused underfitting

The regularized XGBoost model reduced training performance substantially but also worsened validation performance:

| Model | Train MAE | Val MAE | Val R2 |
|---|---:|---:|---:|
| `xgboost_mic_esm2_context` | 0.3545 | 0.4899 | 0.3868 |
| `xgboost_mic_esm2_context_regularized` | 0.4907 | 0.5321 | 0.3101 |

The smaller train-validation gap did not correspond to better generalization. This indicates underfitting rather than successful regularization.

### Combining physicochemical descriptors with ESM2 was best

The best saved validation result came from combining physicochemical descriptors, frozen ESM2 embeddings, and taxonomy/Gram context:

| Model | Val MAE | Val RMSE | Val R2 |
|---|---:|---:|---:|
| `mlp_mic_physchem_mild_regularized` | 0.4028 | 0.5576 | 0.5238 |
| `mlp_mic_physchem_esm2_context_regularized` | 0.3867 | 0.5442 | 0.5519 |

Relative to the physicochemical-only MLP, the combined model improved validation MAE by 0.0161 and validation R2 by 0.0280. The improvement is modest but consistent with the idea that frozen PLM embeddings provide complementary information rather than replacing domain-specific descriptors.

### PCA-compressed ESM2 was competitive but not clearly better

The PCA-compressed ESM2 model reached a validation MAE of 0.3910 and validation R2 of 0.5412:

| Model | Train MAE | Val MAE | Val R2 |
|---|---:|---:|---:|
| `mlp_mic_physchem_esm2_context_regularized` | 0.1849 | 0.3867 | 0.5519 |
| `mlp_mic_physchem_esm2_pca_context_regularized` | 0.1546 | 0.3910 | 0.5412 |
| `mlp_mic_physchem_esm2_pca_context_strong_regularized` | 0.2180 | 0.4023 | 0.5274 |

PCA reduced the ESM2 feature space and preserved competitive validation performance. However, in the saved metrics it did not outperform the non-PCA combined model. The stronger-regularized PCA variant reduced the train-validation gap but also worsened validation error, again suggesting mild underfitting.

## Interpretation

The experiments support three main conclusions.

First, hand-crafted physicochemical descriptors remain strong predictors for this MIC regression task. They outperform frozen ESM2 embeddings when each feature family is used separately.

Second, frozen ESM2 embeddings still add useful information when combined with physicochemical descriptors and organism context. The best combined MLP achieved the lowest validation MAE and highest validation R2 among the saved runs.

Third, the gap between training and validation error should be monitored, but it should not be minimized at the expense of validation performance. The stronger-regularized models had smaller train-validation gaps, but worse validation metrics. For this dataset, the mildly regularized combined MLP appears to preserve useful model capacity while maintaining the best validation result.

## Thesis-Ready Summary

Frozen ESM2 embeddings were evaluated as pretrained protein language model features for MIC regression. Embeddings were generated once using `facebook/esm2_t12_35M_UR50D` and cached to avoid repeated transformer inference. When used alone with organism context, ESM2 features underperformed the existing physicochemical descriptor baseline. However, when ESM2 embeddings were combined with physicochemical descriptors and taxonomy/Gram context in an MLP, validation performance improved from MAE 0.4028 and R2 0.5238 to MAE 0.3867 and R2 0.5519. This suggests that frozen protein language model embeddings provide complementary sequence information, but do not fully replace task-specific physicochemical features for this dataset.

PCA compression of ESM2 embeddings produced competitive but slightly weaker validation performance than the full combined embedding model. Stronger regularization reduced training performance and narrowed the train-validation gap, but did not improve validation error. Therefore, the best current model is `mlp_mic_physchem_esm2_context_regularized`, while `mlp_mic_physchem_esm2_pca_context_regularized` remains a useful lower-dimensional alternative.

## Limitations and Next Steps

These results are based mainly on single-seed runs. Because neural models can vary with random initialization and data ordering, the final thesis comparison should ideally report mean and standard deviation across multiple seeds for the strongest candidates.

Recommended next experiments:

- Repeat the top models with at least 3 seeds: `mlp_mic_physchem_mild_regularized`, `mlp_mic_physchem_esm2_context_regularized`, and `mlp_mic_physchem_esm2_pca_context_regularized`.
- Compare per-phylum and Gram-positive/Gram-negative errors to check whether ESM2 helps specific organism groups.
- Keep the frozen ESM2 setup as the main PLM result before attempting fine-tuning, because it already provides measurable improvement with much lower computational cost.
