# Improvement and Comparative Analysis of Antimicrobial Peptide Prediction Models

## Project Description
Antimicrobial Peptides (AMPs) are short amino acid sequences that play 
an important role in the innate immune system against microorganisms. 
In recent years, they have been actively considered as a potential 
alternative to antibiotics, especially given the spread of 
antibiotic-resistant bacteria.

An important resource for research in this field is **_[DBAASP](https://dbaasp.org/home)_**, 
which contains the amino acid sequences of antimicrobial peptides and their experimentally 
determined biological activity. These data are widely used to create 
various machine learning models that predict a peptide's antimicrobial 
activity based on its sequence.

The goal of the project is to explore possible ways to improve existing 
antimicrobial activity prediction models. For this purpose, data from the 
**_DBAASP_** database will be used to create and train various prediction models. 
Different approaches will also be tested, including modern methods such as 
representing protein sequences using pre-trained models (*Protein Language
Models*).

### The project will involve the following stages:
- Selecting and processing relevant data from the **_DBAASP_** database;
- Creation and training of antimicrobial activity prediction models;
- Experimental evaluation of different approaches;
- Comparison of the resulting models with existing prediction methods;
- Analysis of which methods succeeded in improving predictions and which did not.

The outcome of the project will be a comparative analysis of the prediction 
models. This research will help foster a better understanding of existing 
approaches and identify potential directions for improvement.

### *Project Members:*
- *Zaza Elizbarashvili*
- *Nika Matcharadze*

### *Project Supervisor:*
- *Giorgi Zaalishvili*

# Project structure
```aiignore
amp-prediction/
├── data/
│   ├── raw/                    ← git-ignored, put DBAASP exports here
│   ├── interim/                ← git-ignored, intermediate cleaned files
│   └── processed/
│       ├── splits/             ← ✅ committed (train.csv, val.csv, test.csv)
│       └── embeddings/         ← git-ignored, large cached PLM vectors
├── notebooks/
│   ├── 01_eda/
│   ├── 02_feature_engineering/
│   └── 03_analysis/
├── src/
│   ├── data/
│   │   ├── loader.py           ← load_raw(), load_split()
│   │   └── cleaner.py          ← clean_sequences() (length, non-std AA, dedup)
│   ├── features/
│   │   ├── onehot.py           ← OneHotEncoder
│   │   ├── physicochemical.py  ← PhysicochemicalEncoder (biopython)
│   │   ├── word2vec.py         ← Word2VecEncoder (gensim k-mers)
│   │   └── plm.py              ← PLMEncoder (ESM-2, cached to disk)
│   ├── models/
│   │   ├── base.py             ← BaseModel ABC (fit/predict/predict_proba)
│   │   ├── sklearn_wrapper.py  ← SklearnModel (RF, SVM, LR, GB, KNN)
│   │   ├── mic_runner.py       ← shared MIC baseline training runner
│   │   └── registry.py         ← named model registry
│   ├── evaluation/
│   │   ├── metrics.py          ← compute_metrics() → AUC, MCC, F1, …
│   │   └── plots.py            ← ROC curves, confusion matrix, comparison bar
│   └── utils/
│       ├── config.py           ← load_config() YAML loader + validator
│       ├── seed.py             ← set_seed() (Python, NumPy, PyTorch)
│       └── logger.py           ← get_logger() consistent format
├── experiments/
│   ├── rf_physicochemical.yml  ← Random Forest + physicochemical
│   ├── svm_word2vec.yml        ← SVM + Word2Vec k-mers
│   └── esm2_lr.yml             ← Logistic Regression + ESM-2
├── scripts/
│   └── make_splits.py          ← run once → writes data/processed/splits/
├── run_experiment.py           ← main CLI entry-point (loads config → W&B)
└── env.yml                     ← updated with wandb, biopython, transformers…
```

###### Create an environment with dependencies specified in env.yml:
`conda env create -f env.yml`

`conda env update -n amp -f env.yml --prune`

###### Activate the new environment:
`conda activate amp`

###### Install the project Codex skill:
`mkdir -p ~/.codex/skills && cp -R .codex/skills/amp-prediction-workflows ~/.codex/skills/`

Then use it in Codex prompts with:
`Use $amp-prediction-workflows ...`

###### To deactivate an active environment, use
`conda deactivate`

###### download
`python scripts/fetch_dbaasp_sequences.py`
`python scripts/fetch_dbaasp_cards.py`

###### Build fixed train/val/test splits (needs a negative set first!)
`python scripts/make_splits.py --input dbaasp_raw.csv`

###### 2. Run a classification experiment
`python run_experiment.py --config experiments/rf_physicochemical.yml`

###### Run a MIC regression baseline
`python run_experiment.py --model mic_baseline --input data/processed/splits/train.csv`

###### Run the taxonomy MIC regression baseline
`python run_experiment.py --model taxonomy_mic_baseline --input data/processed/splits/train.csv`

###### Run the XGBoost MIC regression model with sequence descriptors and taxonomy
`python run_experiment.py --model xgboost_mic --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Precompute frozen ESM2 embeddings for MIC regression
`python scripts/make_plm_embeddings.py --input data/processed/amp_mic_activities_taxonomy_features.csv --device mps`

###### Run the XGBoost MIC regression model with frozen ESM2 embeddings and taxonomy
`python run_experiment.py --model xgboost_mic_esm2_context --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Run the XGBoost MIC model with PCA-selected frozen ESM2 embeddings and taxonomy
`python run_experiment.py --model xgboost_mic_esm2_context_selected --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Run the stronger-regularized XGBoost MIC model with frozen ESM2 embeddings and taxonomy
`python run_experiment.py --model xgboost_mic_esm2_context_regularized --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Run the CatBoost MIC regression model with engineered physicochemical features
`python run_experiment.py --model catboost_mic_physchem --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Run the tuned CatBoost MIC regression model
`python run_experiment.py --model catboost_mic_tuned --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Analyze MIC validation errors by Gram, taxonomy, length, MIC range, and duplicates
`python scripts/analyze_mic_errors.py --predictions results/tables/catboost_mic_tuned_predictions.csv --source data/processed/amp_mic_activities_taxonomy_features.csv`

###### Run the PyTorch MLP MIC regression ablation
`python run_experiment.py --model mlp_mic_physchem --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Run the regularized PyTorch MLP MIC regression ablation
`python run_experiment.py --model mlp_mic_physchem_regularized --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Run the mildly regularized PyTorch MLP MIC regression ablation
`python run_experiment.py --model mlp_mic_physchem_mild_regularized --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Run the regularized PyTorch MLP MIC model with frozen ESM2 embeddings and taxonomy
`python run_experiment.py --model mlp_mic_esm2_context_regularized --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Run the regularized PyTorch MLP MIC model with physicochemical and frozen ESM2 features
`python run_experiment.py --model mlp_mic_physchem_esm2_context_regularized --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Run the regularized PyTorch MLP MIC model with physicochemical and PCA-compressed ESM2 features
`python run_experiment.py --model mlp_mic_physchem_esm2_pca_context_regularized --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Run the stronger-regularized PyTorch MLP MIC model with physicochemical and PCA-compressed ESM2 features
`python run_experiment.py --model mlp_mic_physchem_esm2_pca_context_strong_regularized --input data/processed/amp_mic_activities_taxonomy_features.csv`

###### Export the current best MIC model for inference
`python scripts/export_best_mic_model.py`

This writes `results/inference/best_mic_model.joblib` and `results/inference/best_mic_model.manifest.json`. The default exported model is `mlp_mic_physchem_esm2_context_regularized`.

###### Predict MIC values with the exported inference bundle
`python scripts/predict_mic.py --input data/processed/amp_mic_activities_taxonomy_features.csv --output results/inference/mic_predictions.csv`

The input CSV must include `sequence`. If frozen ESM2 features are used, new sequences must already exist in the embedding cache; otherwise run `scripts/make_plm_embeddings.py` on the same input CSV first.

###### 3. Compare all runs visually
Open your Weights & Biases project dashboard.


###### getting features from extracted sequences using modlAMP
`python scripts/extracting_features_from_sequence.py`


###### running UI
`uvicorn app.main:app --host 127.0.0.1 --port 8000`

აქ გამოდის რომ გვაქვს 115K raw, აქედან გვაქვს 16K განსხვავებული სექვენსი, რომლებიც მეორდება targetების მიხედვით. 
ამ კოდით დატას ბოლოში ვამატებ დაუჰენდლავ როუებს და ეგ მისახედი იქნება ბოლოს. უნდა ვიპოვო რა ქმნის პრობლემას და რატომ ვერ იჰნდლება. 
ასევე, unit-ების კონვერტაციისთვის ამ ფორმულას ვიყენებ და ზაალიშვილის უნდა დავადასტურებინო: µM --> µg/ml: ვალუე * len(sequence) * 110 /1000 
-----ეს ჯობია რო პირიქით გადავიყვანოთ. biomni უნდა დავუხმაროთ და მაგან გაგვიკეთოს. 


პრობლემები: 1)კარგად გავიგოთ რა განსხვავებაა activity_measureს და concentrate-ს შორის. რომელი გამოვიყენოთ. 
            2)როცა ერთიდაიგივე სექვენსი გვხვდება ბევრჯერ(სხვადასხვს target species-ზე), მხოლოდ ერთხელ უნდა შევიყვანოთ კლასიფიკაცის დატაში?
            აქ ასევე სავარაუდოდ გვჭირდება target specieების კლასტერიზაცია ან რამე მსგავის, გასავლელია ზაალიშვილთან
            3)ტრნინგი უნდა გავაკეთოთ სხვადასხვა სახეობისთვის ცალცალკე?მაგრამ 5K სახეობაა. 
            4) ჟურნალები არაა ჯეისონში. მხოლოდ არის რეფერენს აიდი. ეგენი წამოვიღე ჯერჯერობით //TODO
            5) ეს დიდი და პატარა ასოები რამე ინფორმაციის მატარებელია?
            6) 
