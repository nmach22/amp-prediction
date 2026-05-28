# Repository Guidelines

## Project Structure & Module Organization
This is a Python project for antimicrobial peptide prediction experiments. Core code lives in `src/`: `data/` loads and cleans DBAASP-derived data, `features/` builds sequence encodings, `models/` wraps estimators, `evaluation/` computes metrics and plots, and `utils/` contains config, logging, and seed helpers. Experiment definitions are YAML files in `experiments/`; `run_experiment.py` is the main CLI entry point.

Data is organized under `data/`: keep raw exports in `data/raw/`, intermediate files in `data/interim/`, fixed splits in `data/processed/splits/`, and embedding caches in `data/processed/embeddings/`. Exploratory notebooks belong in `notebooks/`; one-off data scripts belong in `scripts/`. Generated figures and tables go under `results/figures/` and `results/tables/`.

## Build, Test, and Development Commands
- `conda env create -f env.yml`: create the `amp` environment.
- `conda activate amp`: activate the project environment.
- `python scripts/fetch_dbaasp_sequences.py`: fetch DBAASP sequence data.
- `python scripts/make_splits.py --input dbaasp_raw.csv`: generate train/validation/test splits.
- `python run_experiment.py --config experiments/rf_physicochemical.yml`: run one experiment and log metrics/artifacts to W&B.

## Coding Style & Naming Conventions
Use standard Python style with 4-space indentation, snake_case for functions and variables, and PascalCase for classes such as `PhysicochemicalEncoder` or `SklearnModel`. Keep modules focused and add new encoders, models, or metrics in the matching `src/` subpackage. Use `pathlib.Path` for file paths and YAML configs for experiment parameters.

## Testing Guidelines
No committed test suite is currently present. When adding tests, use `pytest`, place them under `tests/`, and mirror the source layout, for example `tests/features/test_physicochemical.py` or `tests/data/test_loader.py`. Focus on deterministic cleaning, feature shapes, config validation, and metric calculations. Run `pytest` before opening a pull request.

## Commit & Pull Request Guidelines
Recent commit messages are short phrases such as `split data`, `edited notebook`, and `update plots`. Keep future commits concise, but make the subject specific to the changed behavior.

Pull requests should include a short description, affected experiment or data path, commands run, and metric changes when model behavior changes. Include screenshots or saved figure paths for plot/notebook updates. Do not commit large raw data, cached embeddings, virtual environments, IDE files, or generated run artifacts unless required.

## Security & Configuration Tips
Keep credentials and API keys out of the repository. Treat DBAASP exports and generated caches as reproducible artifacts; store large files in ignored data directories and document how to recreate them. Prefer extending `experiments/*.yml` over hard-coding paths, seeds, or model parameters.
