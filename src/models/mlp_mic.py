"""PyTorch MLP MIC regression ablation using physicochemical descriptors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.features.plm import (
    DEFAULT_ESM2_MODEL,
    DEFAULT_MIC_EMBEDDING_PATH,
    embeddings_for_sequences,
    load_embedding_cache,
    load_embedding_cache_metadata,
)
from src.features.sequence_descriptors import SequenceDescriptorEncoder
from src.models.base import BaseModel
from src.models.catboost_mic import (
    CATBOOST_CATEGORICAL_COLUMNS,
    DESCRIPTOR_FEATURE_SET,
    ENGINEERED_FEATURE_COLUMNS,
    TAXONOMY_RANK_COLUMNS,
    build_engineered_physchem_features,
    load_catboost_mic_data,
)
from src.models.taxonomy_mic_baseline import evaluate_taxonomy_predictions


def load_mlp_mic_data(path: str) -> pd.DataFrame:
    """Load cleaned MIC rows for the MLP ablation."""
    return load_catboost_mic_data(path)


def build_mlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build numeric descriptor and one-hot target metadata features."""
    descriptor_encoder = SequenceDescriptorEncoder(feature_set=DESCRIPTOR_FEATURE_SET)
    sequence_features = descriptor_encoder.encode(df["sequence"])
    engineered = build_engineered_physchem_features(sequence_features)
    categoricals = pd.get_dummies(
        _categorical_frame(df),
        columns=list(CATBOOST_CATEGORICAL_COLUMNS),
        prefix=list(CATBOOST_CATEGORICAL_COLUMNS),
        dtype=float,
    )
    return pd.concat(
        [
            sequence_features.reset_index(drop=True),
            engineered.reset_index(drop=True),
            categoricals.reset_index(drop=True),
        ],
        axis=1,
    ).astype(float)


def build_mlp_esm2_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build cached ESM2 embeddings plus one-hot Gram and taxonomy features."""
    embedding_features = _esm2_embedding_features(df)
    context_features = _one_hot_gram_taxonomy_features(df)
    return pd.concat(
        [
            embedding_features.reset_index(drop=True),
            context_features.reset_index(drop=True),
        ],
        axis=1,
    ).astype(float)


def build_mlp_physchem_esm2_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build physicochemical descriptors plus cached ESM2 and taxonomy context."""
    physchem_features = build_mlp_features(df).add_prefix("physchem_")
    embedding_features = _esm2_embedding_features(df)
    context_features = _one_hot_gram_taxonomy_features(df).add_prefix("context_")
    return pd.concat(
        [
            physchem_features.reset_index(drop=True),
            embedding_features.reset_index(drop=True),
            context_features.reset_index(drop=True),
        ],
        axis=1,
    ).astype(float)


def _esm2_embedding_features(df: pd.DataFrame) -> pd.DataFrame:
    embeddings = embeddings_for_sequences(
        df["sequence"].astype(str).tolist(),
        DEFAULT_MIC_EMBEDDING_PATH,
    )
    return pd.DataFrame(
        embeddings,
        columns=[f"esm2_{index}" for index in range(embeddings.shape[1])],
    )


def _one_hot_gram_taxonomy_features(df: pd.DataFrame) -> pd.DataFrame:
    context_columns = ("gram_status", *TAXONOMY_RANK_COLUMNS)
    context = pd.DataFrame(index=df.index)
    for column in context_columns:
        if column in df.columns:
            values = df[column]
        else:
            values = pd.Series("Unknown", index=df.index)
        context[column] = (
            values.fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
    context_features = pd.get_dummies(
        context,
        columns=list(context_columns),
        prefix=list(context_columns),
        dtype=float,
    )
    return context_features


def _categorical_frame(df: pd.DataFrame) -> pd.DataFrame:
    categorical = pd.DataFrame(index=df.index)
    for column in CATBOOST_CATEGORICAL_COLUMNS:
        if column in df.columns:
            values = df[column]
        else:
            values = pd.Series("Unknown", index=df.index)
        categorical[column] = (
            values.fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
    return categorical


class MlpMicRegressor(BaseModel):
    """Regularized PyTorch MLP for log10(MIC)."""

    def __init__(
        self,
        random_state: int = 42,
        hidden_layers: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 300,
        patience: int = 30,
        batch_size: int = 64,
        noise_std: float = 0.0,
    ):
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "torch is required for the mlp_mic_physchem experiment. "
                "Install the project environment from env.yml."
            ) from exc

        self.random_state = random_state
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.noise_std = noise_std
        self._model = None
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        self.training_history_: list[dict[str, float]] = []
        self.best_epoch_: int | None = None
        self.best_train_mae_: float | None = None
        self.best_val_mae_: float | None = None
        self.best_train_val_mae_gap_: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "MlpMicRegressor":
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        self.training_history_ = []
        self.best_epoch_ = None
        self.best_train_mae_ = None
        self.best_val_mae_ = None
        self.best_train_val_mae_gap_ = None
        X_train = self._fit_preprocess(X)
        y_train = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        X_val_array = None
        y_val_array = None
        if X_val is not None and y_val is not None:
            X_val_array = self._transform(X_val)
            y_val_array = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

        self._model = self._build_network(X_train.shape[1])
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.HuberLoss()
        best_state = None
        best_score = np.inf
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            self._train_epoch(X_train, y_train, optimizer, loss_fn)
            train_metrics = self._regression_metrics(X_train, y_train, loss_fn)
            row = {
                "epoch": epoch,
                **{f"train_{name}": value for name, value in train_metrics.items()},
            }
            score = train_metrics["mae"]
            if X_val_array is not None and y_val_array is not None:
                val_metrics = self._regression_metrics(
                    X_val_array,
                    y_val_array,
                    loss_fn,
                )
                row.update(
                    {f"val_{name}": value for name, value in val_metrics.items()}
                )
                score = val_metrics["mae"]
            self.training_history_.append(row)

            if score + 1e-8 < best_score:
                best_score = score
                self.best_epoch_ = epoch
                self.best_train_mae_ = float(train_metrics["mae"])
                self.best_val_mae_ = float(score)
                if X_val_array is not None and y_val_array is not None:
                    self.best_train_val_mae_gap_ = float(
                        val_metrics["mae"] - train_metrics["mae"]
                    )
                else:
                    self.best_train_val_mae_gap_ = None
                best_state = {
                    key: value.detach().clone()
                    for key, value in self._model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fit before predict().")
        import torch

        X_array = self._transform(X)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(self._tensor(X_array)).cpu().numpy().ravel()
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(
            "MlpMicRegressor is a regression model and does not expose "
            "class probabilities."
        )

    def metric_history(self) -> list[dict]:
        """Return per-epoch train/validation metrics for W&B curves."""
        rows = []
        metric_names = ("loss", "mae", "rmse", "r2")
        for history_row in self.training_history_:
            epoch = int(history_row["epoch"])
            for split in ("train", "val"):
                metrics = {
                    name: history_row[f"{split}_{name}"]
                    for name in metric_names
                    if f"{split}_{name}" in history_row
                }
                if not metrics:
                    continue
                rows.append(
                    {
                        "step": epoch,
                        "split": split,
                        "metrics": metrics,
                    }
                )
        return rows

    def artifact_metadata(self, feature_columns: list[str]) -> dict:
        return {
            "mlp_hidden_layers": list(self.hidden_layers),
            "mlp_dropout": self.dropout,
            "mlp_learning_rate": self.learning_rate,
            "mlp_weight_decay": self.weight_decay,
            "mlp_max_epochs": self.max_epochs,
            "mlp_patience": self.patience,
            "mlp_batch_size": self.batch_size,
            "mlp_noise_std": self.noise_std,
            "mlp_best_epoch": self.best_epoch_,
            "mlp_train_mae_at_best_epoch": self.best_train_mae_,
            "mlp_best_validation_mae": self.best_val_mae_,
            "mlp_train_val_mae_gap_at_best_epoch": self.best_train_val_mae_gap_,
            "mlp_training_history": self.training_history_,
            "numeric_imputation_medians": {
                column: float(value)
                for column, value in zip(
                    feature_columns,
                    self._imputer.statistics_,
                    strict=False,
                )
            },
        }

    def _fit_preprocess(self, X: pd.DataFrame) -> np.ndarray:
        raw = X.replace([np.inf, -np.inf], np.nan).astype(float)
        imputed = self._imputer.fit_transform(raw)
        return self._scaler.fit_transform(imputed).astype(np.float32)

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        raw = X.replace([np.inf, -np.inf], np.nan).astype(float)
        imputed = self._imputer.transform(raw)
        return self._scaler.transform(imputed).astype(np.float32)

    def _build_network(self, input_dim: int):
        from torch import nn

        layers = []
        previous = input_dim
        for width in self.hidden_layers:
            layers.append(nn.Linear(previous, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        return nn.Sequential(*layers)

    def _train_epoch(self, X: np.ndarray, y: np.ndarray, optimizer, loss_fn) -> float:
        if self._model is None:
            raise RuntimeError("Model must be initialized before training.")
        import torch

        self._model.train()
        indices = torch.randperm(len(X)).numpy()
        losses = []
        for start in range(0, len(X), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            single_row_batch = len(batch_idx) == 1
            if single_row_batch:
                self._model.eval()
            X_batch = self._tensor(X[batch_idx])
            if not single_row_batch:
                X_batch = self._add_training_noise(X_batch)
            y_batch = self._tensor(y[batch_idx])
            optimizer.zero_grad()
            loss = loss_fn(self._model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            if single_row_batch:
                self._model.train()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def _mae(self, X: np.ndarray, y: np.ndarray) -> float:
        if self._model is None:
            raise RuntimeError("Model must be initialized before evaluation.")
        import torch

        self._model.eval()
        with torch.no_grad():
            pred = self._model(self._tensor(X))
            target = self._tensor(y)
            return float(torch.mean(torch.abs(pred - target)).cpu())

    def _regression_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss_fn,
    ) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Model must be initialized before evaluation.")
        import torch

        self._model.eval()
        with torch.no_grad():
            pred_tensor = self._model(self._tensor(X))
            target_tensor = self._tensor(y)
            loss = float(loss_fn(pred_tensor, target_tensor).cpu())
        y_true = y.ravel().astype(float)
        y_pred = pred_tensor.cpu().numpy().ravel().astype(float)
        abs_error = np.abs(y_pred - y_true)
        return {
            "loss": loss,
            "mae": float(np.mean(abs_error)),
            "rmse": float(np.sqrt(np.mean(np.square(y_pred - y_true)))),
            "r2": _safe_r2(y_true, y_pred),
        }

    def _tensor(self, values: np.ndarray):
        import torch

        return torch.as_tensor(values, dtype=torch.float32)

    def _add_training_noise(self, X_batch):
        if self.noise_std <= 0.0:
            return X_batch
        import torch

        return X_batch + torch.randn_like(X_batch) * self.noise_std


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    denominator = float(np.sum(np.square(y_true - np.mean(y_true))))
    if denominator == 0.0:
        return 0.0
    numerator = float(np.sum(np.square(y_true - y_pred)))
    return 1.0 - numerator / denominator


def build_model(random_state: int = 42) -> MlpMicRegressor:
    """Create the PyTorch MLP MIC regressor."""
    return MlpMicRegressor(random_state=random_state)


def build_regularized_model(random_state: int = 42) -> MlpMicRegressor:
    """Create the regularized PyTorch MLP MIC regressor."""
    return MlpMicRegressor(
        random_state=random_state,
        hidden_layers=(128, 64, 32),
        dropout=0.35,
        learning_rate=5e-4,
        weight_decay=5e-4,
        max_epochs=400,
        patience=20,
        noise_std=0.01,
    )


def build_mild_regularized_model(random_state: int = 42) -> MlpMicRegressor:
    """Create a milder regularized MLP after the stronger variant underfit."""
    return MlpMicRegressor(
        random_state=random_state,
        hidden_layers=(192, 96, 48),
        dropout=0.25,
        learning_rate=7e-4,
        weight_decay=2e-4,
        max_epochs=400,
        patience=25,
        noise_std=0.005,
    )


def build_esm2_context_regularized_model(random_state: int = 42) -> MlpMicRegressor:
    """Create a regularized MLP for dense frozen ESM2 context features."""
    return MlpMicRegressor(
        random_state=random_state,
        hidden_layers=(128, 64),
        dropout=0.4,
        learning_rate=5e-4,
        weight_decay=1e-3,
        max_epochs=400,
        patience=25,
        batch_size=64,
        noise_std=0.01,
    )


def build_physchem_esm2_context_regularized_model(
    random_state: int = 42,
) -> MlpMicRegressor:
    """Create a regularized MLP for combined physicochemical and ESM2 features."""
    return MlpMicRegressor(
        random_state=random_state,
        hidden_layers=(192, 96, 48),
        dropout=0.35,
        learning_rate=5e-4,
        weight_decay=7e-4,
        max_epochs=450,
        patience=30,
        batch_size=64,
        noise_std=0.01,
    )


def build_physchem_esm2_context_strong_regularized_model(
    random_state: int = 42,
) -> MlpMicRegressor:
    """Create a stronger-regularized MLP for combined PCA-ESM2 features."""
    return MlpMicRegressor(
        random_state=random_state,
        hidden_layers=(128, 64, 32),
        dropout=0.4,
        learning_rate=5e-4,
        weight_decay=1e-3,
        max_epochs=450,
        patience=30,
        batch_size=64,
        noise_std=0.01,
    )


def mlp_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return feature metadata stored with the trained artifact."""
    return {
        "sequence_descriptor_columns": SequenceDescriptorEncoder(
            feature_set=DESCRIPTOR_FEATURE_SET
        ).feature_names(),
        "engineered_feature_columns": list(ENGINEERED_FEATURE_COLUMNS),
        "categorical_encoding": "one_hot_target_gram_taxonomy",
        "descriptor_library": "modlamp_plus_reduced_alphabet_kmers",
        "sequence_feature_set": DESCRIPTOR_FEATURE_SET,
        "target": "log10_mic",
        "duplicate_measurements": "median_log_mic_by_sequence_target",
        "null_policy": "taxonomy_unknown_numeric_train_median",
    }


def mlp_esm2_context_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return metadata for frozen ESM2 MLP features."""
    metadata = load_embedding_cache_metadata(DEFAULT_MIC_EMBEDDING_PATH)
    _, embeddings = load_embedding_cache(DEFAULT_MIC_EMBEDDING_PATH)
    return {
        "embedding_model": metadata.get("model_name", DEFAULT_ESM2_MODEL),
        "embedding_path": str(DEFAULT_MIC_EMBEDDING_PATH),
        "embedding_dim": int(embeddings.shape[1]),
        "categorical_encoding": "one_hot_gram_taxonomy",
        "target": "log10_mic",
        "target_features": "frozen_esm2_one_hot_taxonomy_gram",
        "duplicate_measurements": "median_log_mic_by_sequence_target",
        "null_policy": "taxonomy_unknown_one_hot",
    }


def build_mlp_physchem_esm2_genome_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build physicochemical descriptors plus ESM2 plus genome oligo features.

    Replaces one-hot taxonomy/gram context with continuous genome-level
    oligonucleotide composition vectors from GenomeEncoder.
    """
    from src.features.genome import GenomeEncoder

    physchem_features = build_mlp_features(df).add_prefix("physchem_")
    embedding_features = _esm2_embedding_features(df)

    genome_encoder = GenomeEncoder()
    genome_vectors = genome_encoder.encode(df["target_activity_name"])
    genome_df = pd.DataFrame(
        genome_vectors,
        columns=[f"genome_{name}" for name in genome_encoder.feature_names()],
    )

    return pd.concat(
        [
            physchem_features.reset_index(drop=True),
            embedding_features.reset_index(drop=True),
            genome_df.reset_index(drop=True),
        ],
        axis=1,
    ).astype(float)


def pca_reduce_esm2_and_genome_features(
    X_train: pd.DataFrame,
    validation_features: dict[str, pd.DataFrame],
    y_train: np.ndarray,
    esm2_n_components: int = 128,
    genome_n_components: int = 64,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """PCA-reduce both ESM2 and genome feature blocks on train set only."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    del y_train

    esm2_cols = [c for c in X_train.columns if c.startswith("esm2_")]
    genome_cols = [c for c in X_train.columns if c.startswith("genome_")]
    passthrough_cols = [
        c for c in X_train.columns if c not in esm2_cols and c not in genome_cols
    ]

    # ESM2 PCA
    esm2_components = min(esm2_n_components, len(esm2_cols), len(X_train))
    esm2_scaler = StandardScaler()
    esm2_pca = PCA(n_components=esm2_components, random_state=42)
    train_esm2_scaled = esm2_scaler.fit_transform(X_train[esm2_cols].astype(float))
    train_esm2_pca = esm2_pca.fit_transform(train_esm2_scaled)

    # Genome PCA
    genome_components = min(genome_n_components, len(genome_cols), len(X_train))
    genome_scaler = StandardScaler()
    genome_pca = PCA(n_components=genome_components, random_state=42)
    train_genome_scaled = genome_scaler.fit_transform(
        X_train[genome_cols].astype(float)
    )
    train_genome_pca = genome_pca.fit_transform(train_genome_scaled)

    X_train_reduced = _join_esm2_genome_pca(
        train_esm2_pca, train_genome_pca,
        X_train[passthrough_cols],
        esm2_components, genome_components,
    )

    transformed_validation = {}
    for split_name, X_split in validation_features.items():
        split_esm2 = esm2_scaler.transform(X_split[esm2_cols].astype(float))
        split_esm2_pca = esm2_pca.transform(split_esm2)
        split_genome = genome_scaler.transform(X_split[genome_cols].astype(float))
        split_genome_pca = genome_pca.transform(split_genome)
        transformed_validation[split_name] = _join_esm2_genome_pca(
            split_esm2_pca, split_genome_pca,
            X_split[passthrough_cols],
            esm2_components, genome_components,
        )

    metadata = {
        "feature_transform": "train_only_pca_on_esm2_and_genome",
        "esm2_original_dim": len(esm2_cols),
        "esm2_pca_components": esm2_components,
        "esm2_pca_explained_variance_total": float(
            np.sum(esm2_pca.explained_variance_ratio_)
        ),
        "genome_original_dim": len(genome_cols),
        "genome_pca_components": genome_components,
        "genome_pca_explained_variance_total": float(
            np.sum(genome_pca.explained_variance_ratio_)
        ),
        "passthrough_feature_columns": passthrough_cols,
    }
    return X_train_reduced, transformed_validation, metadata


def _join_esm2_genome_pca(
    esm2_pca_values: np.ndarray,
    genome_pca_values: np.ndarray,
    passthrough: pd.DataFrame,
    esm2_n: int,
    genome_n: int,
) -> pd.DataFrame:
    esm2_df = pd.DataFrame(
        esm2_pca_values,
        columns=[f"esm2_pca_{i}" for i in range(esm2_n)],
        index=passthrough.index,
    )
    genome_df = pd.DataFrame(
        genome_pca_values,
        columns=[f"genome_pca_{i}" for i in range(genome_n)],
        index=passthrough.index,
    )
    return pd.concat(
        [
            esm2_df.reset_index(drop=True),
            genome_df.reset_index(drop=True),
            passthrough.reset_index(drop=True).astype(float),
        ],
        axis=1,
    )


def build_physchem_esm2_genome_regularized_model(
    random_state: int = 42,
) -> MlpMicRegressor:
    """Create a regularized MLP for physchem + PCA-ESM2 + genome oligo features."""
    return MlpMicRegressor(
        random_state=random_state,
        hidden_layers=(192, 96, 48),
        dropout=0.35,
        learning_rate=5e-4,
        weight_decay=7e-4,
        max_epochs=450,
        patience=30,
        batch_size=64,
        noise_std=0.01,
    )


def mlp_physchem_esm2_genome_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return metadata for combined physicochemical, ESM2, and genome features."""
    metadata = load_embedding_cache_metadata(DEFAULT_MIC_EMBEDDING_PATH)
    _, embeddings = load_embedding_cache(DEFAULT_MIC_EMBEDDING_PATH)
    return {
        "embedding_model": metadata.get("model_name", DEFAULT_ESM2_MODEL),
        "embedding_path": str(DEFAULT_MIC_EMBEDDING_PATH),
        "embedding_dim": int(embeddings.shape[1]),
        "sequence_descriptor_columns": SequenceDescriptorEncoder(
            feature_set=DESCRIPTOR_FEATURE_SET
        ).feature_names(),
        "engineered_feature_columns": list(ENGINEERED_FEATURE_COLUMNS),
        "descriptor_library": "modlamp_plus_reduced_alphabet_kmers",
        "sequence_feature_set": DESCRIPTOR_FEATURE_SET,
        "genome_features": "oligonucleotide_k3_k4_k5_ddh_gyrb",
        "genome_feature_dim": 1364,
        "target": "log10_mic",
        "target_features": "physicochemical_engineered_pca_frozen_esm2_genome_oligo",
        "duplicate_measurements": "median_log_mic_by_sequence_target",
    }


def mlp_physchem_esm2_context_artifact_metadata(df: pd.DataFrame) -> dict:
    """Return metadata for combined physicochemical and frozen ESM2 features."""
    metadata = mlp_esm2_context_artifact_metadata(df)
    metadata.update(
        {
            "sequence_descriptor_columns": SequenceDescriptorEncoder(
                feature_set=DESCRIPTOR_FEATURE_SET
            ).feature_names(),
            "engineered_feature_columns": list(ENGINEERED_FEATURE_COLUMNS),
            "descriptor_library": "modlamp_plus_reduced_alphabet_kmers",
            "sequence_feature_set": DESCRIPTOR_FEATURE_SET,
            "target_features": (
                "physicochemical_engineered_frozen_esm2_one_hot_taxonomy_gram"
            ),
        }
    )
    return metadata


__all__ = [
    "MlpMicRegressor",
    "build_esm2_context_regularized_model",
    "build_mlp_esm2_context_features",
    "build_mlp_features",
    "build_mlp_physchem_esm2_context_features",
    "build_mlp_physchem_esm2_genome_features",
    "build_mild_regularized_model",
    "build_model",
    "build_physchem_esm2_context_regularized_model",
    "build_physchem_esm2_context_strong_regularized_model",
    "build_physchem_esm2_genome_regularized_model",
    "build_regularized_model",
    "evaluate_taxonomy_predictions",
    "load_mlp_mic_data",
    "mlp_esm2_context_artifact_metadata",
    "mlp_artifact_metadata",
    "mlp_physchem_esm2_context_artifact_metadata",
    "mlp_physchem_esm2_genome_artifact_metadata",
    "pca_reduce_esm2_and_genome_features",
]
