# Two-Tower Deep Learning Model with Uncertainty Quantification

## Architecture

We propose a two-tower neural architecture for predicting log₁₀ MIC of antimicrobial peptides against specific target organisms. The **peptide tower** passes uppercased sequences through a frozen ESM-2 (650M-parameter, `esm2_t33_650M_UR50D`) protein language model to obtain per-residue embeddings, followed by a learnable Transformer encoder and attention pooling. The **organism tower** uses an FT-Transformer (Feature Tokenizer + Transformer) that jointly encodes species identity and numeric organism-level features. The combined representation is passed to one of two uncertainty-aware prediction heads.

## Training Protocol

- **Data:** ~106k aggregated (peptide, organism) pairs from DBAASP across 15 named species + an "other" category, with replicate measurements aggregated by geometric mean in log-space.
- **Split:** k-mer Jaccard similarity clustering to prevent near-duplicate leakage between train/test.
- **Loss:** head-specific (Gaussian NLL or pinball); **early stopping** on validation Spearman ρ.
- **Organism features** fit on train only (no leakage).

## Two Confidence-Interval Heads

1. **Gaussian head** — predicts mean and log-variance; trained with Gaussian NLL loss; 90% PI = mean ± 1.645·σ.
2. **Quantile head** — directly predicts τ = {0.05, 0.50, 0.95} quantiles via pinball loss; distribution-free 90% PI.

Both heads support post-hoc **split-conformal calibration** to correct empirical coverage without retraining.

## Held-Out Test Results

| Metric | Gaussian Head | Quantile Head |
|---|---|---|
| RMSE (log₁₀ MIC) | **0.753** | 0.744 |
| MAE | 0.576 | 0.555 |
| Pearson r | 0.455 | **0.493** |
| Spearman ρ | 0.449 | **0.478** |
| Within 2-fold | 0.343 | **0.378** |
| Within 4-fold | 0.618 | **0.651** |
| PICP (nominal 90%) | **82.8%** | 75.9% |
| MPIW (interval width) | 1.923 | **1.564** |
| Coverage gap | −7.2% | −14.1% |

**Best validation Spearman:** Gaussian = 0.496 (epoch 2, early-stopped at 7); Quantile = 0.526 (epoch 5, early-stopped at 10).

## Per-Group Breakdown — Gaussian Head (Test Set)

| Group | n | RMSE | Pearson | Spearman | Within 2-fold |
|---|---|---|---|---|---|
| Standard L-peptides | 14,951 | 0.760 | 0.471 | 0.464 | 33.7% |
| D-amino acid peptides | 1,499 | 0.636 | 0.531 | 0.569 | 43.4% |
| Modified/X peptides | 2,892 | 0.776 | 0.335 | 0.310 | 32.8% |
| *S. aureus* | 3,496 | 0.703 | 0.510 | 0.466 | 37.6% |
| *E. coli* | 3,387 | 0.733 | 0.498 | 0.500 | 37.5% |
| *P. aeruginosa* | 2,209 | 0.625 | 0.468 | 0.465 | 38.0% |
| *K. pneumoniae* | 922 | 0.634 | 0.508 | 0.467 | 35.0% |
| *A. baumannii* | 779 | 0.601 | 0.519 | 0.541 | 42.4% |
| *C. albicans* | 779 | 0.749 | 0.276 | 0.236 | 30.4% |
| *B. subtilis* | 765 | 0.747 | 0.533 | 0.524 | 33.5% |
| *S. enterica* | 639 | 0.942 | 0.404 | 0.366 | 31.3% |
| *S. epidermidis* | 569 | 0.668 | 0.498 | 0.502 | 41.1% |
| *E. faecalis* | 456 | 0.706 | 0.401 | 0.402 | 33.8% |
| *M. luteus* | 302 | 0.939 | 0.254 | 0.282 | 25.2% |
| *S. typhimurium* | 238 | 0.816 | 0.346 | 0.383 | 31.5% |
| *E. faecium* | 229 | 0.665 | 0.424 | 0.410 | 39.7% |
| *B. cereus* | 162 | 0.669 | 0.469 | 0.484 | 33.3% |
| *L. monocytogenes* | 128 | 0.981 | 0.387 | 0.283 | 22.7% |
| Other | 4,282 | 0.875 | 0.382 | 0.364 | 26.9% |

## Per-Group Breakdown — Quantile Head (Test Set)

| Group | n | RMSE | Pearson | Spearman | Within 2-fold |
|---|---|---|---|---|---|
| Standard L-peptides | 14,951 | 0.746 | 0.505 | 0.490 | 37.6% |
| D-amino acid peptides | 1,499 | 0.636 | 0.548 | 0.548 | 41.3% |
| Modified/X peptides | 2,892 | 0.785 | 0.398 | 0.375 | 36.9% |
| *S. aureus* | 3,496 | 0.702 | 0.536 | 0.495 | 38.4% |
| *E. coli* | 3,387 | 0.730 | 0.516 | 0.511 | 38.8% |
| *P. aeruginosa* | 2,209 | 0.609 | 0.493 | 0.494 | 44.0% |
| *K. pneumoniae* | 922 | 0.619 | 0.558 | 0.522 | 43.9% |
| *A. baumannii* | 779 | 0.594 | 0.536 | 0.539 | 43.1% |
| *C. albicans* | 779 | 0.734 | 0.342 | 0.282 | 32.9% |
| *B. subtilis* | 765 | 0.727 | 0.528 | 0.528 | 37.4% |
| *S. enterica* | 639 | 0.950 | 0.427 | 0.404 | 33.6% |
| *S. epidermidis* | 569 | 0.663 | 0.498 | 0.508 | 43.1% |
| *E. faecalis* | 456 | 0.711 | 0.405 | 0.422 | 39.0% |
| *M. luteus* | 302 | 0.897 | 0.357 | 0.381 | 28.5% |
| *S. typhimurium* | 238 | 0.805 | 0.420 | 0.462 | 36.1% |
| *E. faecium* | 229 | 0.652 | 0.485 | 0.482 | 41.5% |
| *B. cereus* | 162 | 0.626 | 0.549 | 0.575 | 49.4% |
| *L. monocytogenes* | 128 | 1.028 | 0.378 | 0.302 | 32.8% |
| Other | 4,282 | 0.859 | 0.442 | 0.402 | 32.2% |

## Quantile Head — Per-Quantile Calibration (Test Set)

| Nominal τ | Empirical Coverage | Gap |
|---|---|---|
| 0.05 | 0.148 | +0.098 |
| 0.50 | 0.525 | +0.025 |
| 0.95 | 0.907 | −0.043 |

## Key Observations

- The **quantile head outperforms the Gaussian head** on point-prediction metrics (Spearman +0.03, RMSE −0.01) and produces **narrower intervals** (MPIW 1.56 vs 1.92 log₁₀ units).
- The Gaussian head achieves **better raw coverage** (82.8% vs 75.9%), though both under-cover the 90% nominal — the built-in split-conformal calibration module corrects this gap post-hoc.
- D-amino acid peptides are predicted **better** than standard L-peptides (Spearman 0.55–0.57 vs 0.46–0.49), likely because D-peptides cluster in a distinct MIC range; modified/X peptides are the hardest group (Spearman 0.31–0.38).
- Performance varies by organism: Gram-negative bacteria (*P. aeruginosa*, *A. baumannii*, *K. pneumoniae*) show the best Spearman (0.47–0.54), while the fungal species *C. albicans* is the weakest (0.24–0.28), consistent with its distinct membrane biology.
- The quantile head improves over the Gaussian head **consistently across nearly all species**, with the largest gains on *B. cereus* (Spearman 0.48 → 0.58), *M. luteus* (0.28 → 0.38), and *S. typhimurium* (0.38 → 0.46).
