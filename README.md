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

# Project sructure
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
│   │   └── sklearn_wrapper.py  ← SklearnModel (RF, SVM, LR, GB, KNN)
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
├── run_experiment.py           ← main CLI entry-point (loads config → MLflow)
└── env.yml                     ← updated with mlflow, biopython, transformers…
```

###### Create an environment with dependencies specified in env.yml:
`conda env create -f env.yml`

###### Activate the new environment:
`conda activate amp`

###### To deactivate an active environment, use
`conda deactivate`

###### download
`python scripts/fetch_dbaasp_sequences.py`
`python scripts/fetch_dbaasp_cards.py`

###### Build fixed train/val/test splits (needs a negative set first!)
`python scripts/make_splits.py --input dbaasp_raw.csv`

###### 2. Run any experiment
`python run_experiment.py --config experiments/rf_physicochemical.yml`

###### 3. Compare all runs visually
`mlflow ui`


###### extracting sequences with MICs from JSON
`python scripts/extracting_sequences_from_full_data.py`

###### extracting sequences anits features from JSON
`python scripts/extracting_sequences_with_features_from_JSON.py`

###### getting features from extracted sequences using modlAMP
`python scripts/extracting_features_from_sequence.py`


აქ გამოდის რომ გვაქვს 115K raw, აქედან გვაქვს 16K განსხვავებული სექვენსი, რომლებიც მეორდება targetების მიხედვით. 
ამ კოდით დატას ბოლოში ვამატებ დაუჰენდლავ როუებს და ეგ მისახედი იქნება ბოლოს. უნდა ვიპოვო რა ქმნის პრობლემას და რატომ ვერ იჰნდლება. 
ასევე, unit-ების კონვერტაციისთვის ამ ფორმულას ვიყენებ და ზაალიშვილის უნდა დავადასტურებინო: µM --> µg/ml: ვალუე * len(sequence) * 110 /1000 



პრობლემები: 1)კარგად გავიგოთ რა განსხვავებაა activity_measureს და concentrate-ს შორის. რომელი გამოვიყენოთ. 
            2)როცა ერთიდაიგივე სექვენსი გვხვდება ბევრჯერ(სხვადასხვს target species-ზე), მხოლოდ ერთხელ უნდა შევიყვანოთ კლასიფიკაცის დატაში?
            აქ ასევე სავარაუდოდ გვჭირდება target specieების კლასტერიზაცია ან რამე მსგავის, გასავლელია ზაალიშვილთან
            3)ტრნინგი უნდა გავაკეთოთ სხვადასხვა სახეობისთვის ცალცალკე?მაგრამ 5K სახეობაა. 
            4) ჟურნალები არაა ჯეისონში. მხოლოდ არის რეფერენს აიდი. ეგენი წამოვიღე ჯერჯერობით
            5) ეს დიდი და პატარა ასოები რამე ინფორმაციის მატარებელია?