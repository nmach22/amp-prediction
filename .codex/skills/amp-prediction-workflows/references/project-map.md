# Project Map

Use this file to orient quickly in the AMP prediction repository.

## Entry Points

- `run_experiment.py`: main CLI. Exactly one of `--config` or `--model` is required.
- `experiments/*.yml`: YAML-driven binary AMP classification configs.
- `src/models/registry.py`: named MIC regression experiment registry.
- `src/models/mic_runner.py`: shared MIC training, validation, metrics, predictions, and artifact writing.
- `evaluate_saved_model.py`: saved model evaluation helper.

## Source Layout

- `src/data/`: raw DBAASP loaders and sequence cleaning.
- `src/features/`: one-hot, physicochemical, PLM, Word2Vec, taxonomy, and modlamp sequence descriptors.
- `src/models/`: model wrappers, MIC baselines, XGBoost MIC variants, registry, and shared runner.
- `src/evaluation/`: classification metrics and plots.
- `src/utils/`: config, logging, seed, and W&B helpers.
- `scripts/`: data fetching, parsing, Gram classification, split creation, and one-off feature extraction scripts.
- `tests/`: pytest coverage for runner behavior, MIC models, taxonomy features, and sequence descriptors.

## Data And Output Paths

- `data/raw/`: DBAASP exports and fetched raw artifacts.
- `data/interim/`: intermediate processing files.
- `data/processed/splits/`: fixed classification train/val/test CSVs.
- `data/processed/embeddings/`: large PLM embedding caches.
- `results/models/`: trained joblib artifacts.
- `results/tables/`: metrics and predictions CSVs.
- `results/figures/`: plots.

## Canonical Commands

```bash
conda env create -f env.yml
conda activate amp
python scripts/fetch_dbaasp_sequences.py
python scripts/make_splits.py --input dbaasp_raw.csv
python run_experiment.py --config experiments/rf_physicochemical.yml
python run_experiment.py --model mic_baseline --input data/processed/splits/train.csv
python run_experiment.py --model taxonomy_mic_baseline --input data/processed/splits/train.csv
python run_experiment.py --model xgboost_mic --input data/processed/amp_mic_activities_taxonomy_features.csv
pytest
```

Use `--disable-wandb` for local smoke tests when W&B credentials or network access are not needed.
