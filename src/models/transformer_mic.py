"""
Transformer-based MIC regression model with cross-attention between
peptide (ESM-2) and bacteria (oligonucleotide composition) embeddings.

Architecture:
    Peptide branch:  ESM-2 embedding (480d) → projection → peptide tokens
    Bacteria branch: Oligo composition (1344d) → projection → bacteria tokens
    Cross-attention: peptide attends to bacteria, bacteria attends to peptide
    Fusion:          concatenate [CLS] tokens → MLP head → log10(MIC)

Usage:
    python run_experiment.py --model genome_transformer_mic \\
        --input data/raw/amp_mic_activities.csv
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.features.genome import GenomeEncoder
from src.features.plm import (
    DEFAULT_MIC_EMBEDDING_PATH,
    PLMEncoder,
    DEFAULT_ESM2_MODEL,
    load_embedding_cache,
    save_embedding_cache,
)
from src.models.base import BaseModel
from src.models.taxonomy_mic_baseline import evaluate_taxonomy_predictions


# ─── Data Loading ──────────────────────────────────────────────────────────────


def load_transformer_mic_data(path: str) -> pd.DataFrame:
    """Load raw MIC data — only requires sequence, target_activity_name, activity."""
    df = pd.read_csv(path)
    required = {"sequence", "target_activity_name", "activity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["sequence", "target_activity_name", "activity"])
    cleaned["sequence"] = cleaned["sequence"].astype(str).str.upper().str.strip()
    cleaned["target_activity_name"] = cleaned["target_activity_name"].astype(str).str.strip()
    cleaned["activity"] = pd.to_numeric(cleaned["activity"], errors="coerce")
    cleaned = cleaned.dropna(subset=["activity"])
    cleaned = cleaned[cleaned["activity"] > 0]
    cleaned = cleaned[cleaned["sequence"].str.len() > 0]
    cleaned = cleaned[~cleaned["sequence"].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", flags=re.IGNORECASE)]
    cleaned["log_mic"] = np.log10(cleaned["activity"])
    return cleaned.reset_index(drop=True)


# ─── Feature Building ──────────────────────────────────────────────────────────


def _get_or_compute_esm2_embeddings(
    sequences: list[str],
    cache_path: Path = DEFAULT_MIC_EMBEDDING_PATH,
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """Get ESM-2 embeddings, computing missing ones on-the-fly."""
    try:
        cached_seqs, cached_embs = load_embedding_cache(cache_path)
        cache_index = {seq: i for i, seq in enumerate(cached_seqs)}
    except (FileNotFoundError, Exception):
        cached_seqs, cached_embs = [], np.empty((0, 480))
        cache_index = {}

    # Find missing sequences
    missing_seqs = [s for s in set(sequences) if s not in cache_index]

    if missing_seqs:
        print(f"  Computing ESM-2 embeddings for {len(missing_seqs)} new sequences...")
        encoder = PLMEncoder(model_name=DEFAULT_ESM2_MODEL, device=device)
        new_embeddings = encoder.encode(missing_seqs, batch_size=batch_size)

        # Update cache
        all_seqs = list(cached_seqs) + missing_seqs
        all_embs = np.vstack([cached_embs, new_embeddings]) if len(cached_embs) > 0 else new_embeddings
        save_embedding_cache(
            cache_path, all_seqs, all_embs, model_name=DEFAULT_ESM2_MODEL
        )
        print(f"  Updated cache: {len(all_seqs)} total sequences")
        cache_index = {seq: i for i, seq in enumerate(all_seqs)}
        cached_embs = all_embs

    # Return in order
    return np.vstack([cached_embs[cache_index[seq]] for seq in sequences])


def build_transformer_features(df: pd.DataFrame, genome_encoder: GenomeEncoder) -> dict:
    """
    Build feature arrays for the transformer model.
    Returns dict with 'peptide_features' and 'genome_features' arrays.
    """
    # Peptide: ESM-2 embeddings
    sequences = df["sequence"].astype(str).tolist()
    peptide_embs = _get_or_compute_esm2_embeddings(sequences)

    # Bacteria: oligo composition
    target_names = df["target_activity_name"]
    genome_embs = genome_encoder.encode(target_names)

    return {
        "peptide_features": peptide_embs.astype(np.float32),
        "genome_features": genome_embs.astype(np.float32),
    }


# ─── Transformer Architecture ─────────────────────────────────────────────────


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for token sequences."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding[:, :x.size(1), :]


class CrossAttentionBlock(nn.Module):
    """Cross-attention: query attends to key-value from another modality."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-attention
        attn_out, _ = self.cross_attn(query, context, context)
        x = self.norm1(query + attn_out)
        # Feed-forward
        x = self.norm2(x + self.ffn(x))
        return x


class SelfAttentionBlock(nn.Module):
    """Standard self-attention transformer block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class PeptideGenomeTransformer(nn.Module):
    """
    Cross-attention transformer for peptide–bacteria MIC prediction.

    Architecture:
        1. Project peptide (480d) and genome (1344d) into shared d_model space
        2. Chunk each into token sequences (simulating multi-token input)
        3. Self-attention within each modality
        4. Cross-attention: peptide ↔ bacteria interaction
        5. Pool [CLS] tokens → regression head
    """

    def __init__(
        self,
        peptide_dim: int = 480,
        genome_dim: int = 1344,
        d_model: int = 256,
        n_heads: int = 8,
        n_self_layers: int = 2,
        n_cross_layers: int = 2,
        n_tokens_peptide: int = 8,
        n_tokens_genome: int = 16,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tokens_peptide = n_tokens_peptide
        self.n_tokens_genome = n_tokens_genome

        # Project raw features into token sequences
        # Peptide: 480 → n_tokens_peptide tokens of d_model dims
        self.peptide_projection = nn.Sequential(
            nn.Linear(peptide_dim, n_tokens_peptide * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Genome: 1344 → n_tokens_genome tokens of d_model dims
        self.genome_projection = nn.Sequential(
            nn.Linear(genome_dim, n_tokens_genome * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # CLS tokens (learnable)
        self.cls_peptide = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_genome = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_peptide = PositionalEncoding(d_model, n_tokens_peptide + 1)
        self.pos_genome = PositionalEncoding(d_model, n_tokens_genome + 1)

        # Self-attention blocks
        self.self_attn_peptide = nn.ModuleList([
            SelfAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_self_layers)
        ])
        self.self_attn_genome = nn.ModuleList([
            SelfAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_self_layers)
        ])

        # Cross-attention blocks (peptide attends to genome, then genome to peptide)
        self.cross_attn_p2g = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])
        self.cross_attn_g2p = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights for Linear, Conv, Recurrent, Normalization,
        and Embedding layers using recommended schemes.
        """
        for module in self.modules():
            # 1. Linear / Dense Layers
            if isinstance(module, nn.Linear):
                # Kaiming Normal for ReLU/LeakyReLU; use xavier_uniform_ if using Sigmoid/Tanh
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            # 2. Convolutional Layers
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            # 3. Normalization Layers (BatchNorm, LayerNorm, GroupNorm)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                     nn.LayerNorm, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            # 4. Embedding Layers
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

            # 5. Recurrent Layers (LSTM, GRU)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
                        # Optional: Set LSTM forget gate bias to 1.0
                        if isinstance(module, nn.LSTM):
                            n = param.size(0)
                            param.data[n // 4: n // 2].fill_(1.0)

    def forward(
        self,
        peptide_features: torch.Tensor,
        genome_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = peptide_features.size(0)

        # Project to token sequences
        pep_tokens = self.peptide_projection(peptide_features)
        pep_tokens = pep_tokens.view(batch_size, self.n_tokens_peptide, self.d_model)

        gen_tokens = self.genome_projection(genome_features)
        gen_tokens = gen_tokens.view(batch_size, self.n_tokens_genome, self.d_model)

        # Prepend CLS tokens
        cls_p = self.cls_peptide.expand(batch_size, -1, -1)
        cls_g = self.cls_genome.expand(batch_size, -1, -1)
        pep_tokens = torch.cat([cls_p, pep_tokens], dim=1)
        gen_tokens = torch.cat([cls_g, gen_tokens], dim=1)

        # Add positional encoding
        pep_tokens = self.pos_peptide(pep_tokens)
        gen_tokens = self.pos_genome(gen_tokens)

        # Self-attention within each modality
        for layer in self.self_attn_peptide:
            pep_tokens = layer(pep_tokens)
        for layer in self.self_attn_genome:
            gen_tokens = layer(gen_tokens)

        # Cross-attention: peptide ↔ genome
        for p2g, g2p in zip(self.cross_attn_p2g, self.cross_attn_g2p):
            pep_tokens = p2g(pep_tokens, gen_tokens)
            gen_tokens = g2p(gen_tokens, pep_tokens)

        # Pool CLS tokens
        pep_cls = pep_tokens[:, 0, :]  # (batch, d_model)
        gen_cls = gen_tokens[:, 0, :]  # (batch, d_model)

        # Fuse and predict
        fused = torch.cat([pep_cls, gen_cls], dim=1)  # (batch, 2*d_model)
        return self.head(fused)


# ─── Model Wrapper ─────────────────────────────────────────────────────────────


class GenomeTransformerMicRegressor(BaseModel):
    """
    Cross-attention transformer for MIC regression.

    Combines ESM-2 peptide embeddings with oligonucleotide genome features
    through cross-attention layers before regression.
    """

    def __init__(
        self,
        random_state: int = 42,
        d_model: int = 256,
        n_heads: int = 8,
        n_self_layers: int = 2,
        n_cross_layers: int = 2,
        n_tokens_peptide: int = 8,
        n_tokens_genome: int = 16,
        dropout: float = 0.15,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 300,
        patience: int = 35,
        batch_size: int = 128,
        warmup_steps: int = 500,
        device: str = "cpu",
    ):
        self.random_state = random_state
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_self_layers = n_self_layers
        self.n_cross_layers = n_cross_layers
        self.n_tokens_peptide = n_tokens_peptide
        self.n_tokens_genome = n_tokens_genome
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.device = device

        self._model: PeptideGenomeTransformer | None = None
        self._peptide_scaler = StandardScaler()
        self._genome_scaler = StandardScaler()
        self._genome_imputer = SimpleImputer(strategy="constant", fill_value=0.0)

        self.training_history_: list[dict] = []
        self.best_epoch_: int | None = None
        self.best_val_mae_: float | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "GenomeTransformerMicRegressor":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # X is a DataFrame with columns: peptide_0..peptide_479, genome_0..genome_1343
        peptide_cols = [c for c in X.columns if c.startswith("peptide_")]
        genome_cols = [c for c in X.columns if c.startswith("genome_")]

        X_pep_train = self._peptide_scaler.fit_transform(X[peptide_cols].values).astype(np.float32)
        X_gen_train = self._genome_scaler.fit_transform(
            self._genome_imputer.fit_transform(X[genome_cols].values)
        ).astype(np.float32)
        y_train = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        peptide_dim = X_pep_train.shape[1]
        genome_dim = X_gen_train.shape[1]

        # Validation data
        X_pep_val, X_gen_val, y_val_arr = None, None, None
        if X_val is not None and y_val is not None:
            X_pep_val = self._peptide_scaler.transform(X_val[peptide_cols].values).astype(np.float32)
            X_gen_val = self._genome_scaler.transform(
                self._genome_imputer.transform(X_val[genome_cols].values)
            ).astype(np.float32)
            y_val_arr = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

        # Build model
        self._model = PeptideGenomeTransformer(
            peptide_dim=peptide_dim,
            genome_dim=genome_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_self_layers=self.n_self_layers,
            n_cross_layers=self.n_cross_layers,
            n_tokens_peptide=self.n_tokens_peptide,
            n_tokens_genome=self.n_tokens_genome,
            dropout=self.dropout,
        )
        self._model.to(self.device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing with warmup
        total_steps = self.max_epochs * (len(X_pep_train) // self.batch_size + 1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=min(self.warmup_steps / total_steps, 0.1),
            anneal_strategy="cos",
        )

        loss_fn = nn.HuberLoss()
        best_state = None
        best_score = np.inf
        epochs_no_improve = 0

        n_params = sum(p.numel() for p in self._model.parameters())
        print(f"  Transformer params: {n_params:,} | "
              f"peptide_dim={peptide_dim}, genome_dim={genome_dim}, "
              f"d_model={self.d_model}")

        for epoch in range(1, self.max_epochs + 1):
            # Training
            self._model.train()
            indices = torch.randperm(len(X_pep_train)).numpy()
            epoch_losses = []

            for start in range(0, len(X_pep_train), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                pep_batch = torch.tensor(X_pep_train[batch_idx], device=self.device)
                gen_batch = torch.tensor(X_gen_train[batch_idx], device=self.device)
                y_batch = torch.tensor(y_train[batch_idx], device=self.device)

                optimizer.zero_grad()
                pred = self._model(pep_batch, gen_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_losses.append(loss.item())

            # Evaluation
            train_mae = self._eval_mae(X_pep_train, X_gen_train, y_train)
            row = {"epoch": epoch, "train_loss": np.mean(epoch_losses), "train_mae": train_mae}

            score = train_mae
            if X_pep_val is not None:
                val_mae = self._eval_mae(X_pep_val, X_gen_val, y_val_arr)
                row["val_mae"] = val_mae
                score = val_mae

            self.training_history_.append(row)

            if score < best_score - 1e-5:
                best_score = score
                self.best_epoch_ = epoch
                self.best_val_mae_ = float(score)
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epochs_no_improve == 0:
                val_str = f", val_mae={row.get('val_mae', 'N/A'):.4f}" if 'val_mae' in row else ""
                print(f"    Epoch {epoch}: train_mae={train_mae:.4f}{val_str}")

            if epochs_no_improve >= self.patience:
                print(f"  Early stopping at epoch {epoch} (best: {self.best_epoch_})")
                break

        if best_state:
            self._model.load_state_dict(best_state)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fit before predict()")

        peptide_cols = [c for c in X.columns if c.startswith("peptide_")]
        genome_cols = [c for c in X.columns if c.startswith("genome_")]

        X_pep = self._peptide_scaler.transform(X[peptide_cols].values).astype(np.float32)
        X_gen = self._genome_scaler.transform(
            self._genome_imputer.transform(X[genome_cols].values)
        ).astype(np.float32)

        self._model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(X_pep), self.batch_size):
                pep_batch = torch.tensor(X_pep[start:start + self.batch_size], device=self.device)
                gen_batch = torch.tensor(X_gen[start:start + self.batch_size], device=self.device)
                pred = self._model(pep_batch, gen_batch).cpu().numpy()
                preds.append(pred)

        return np.concatenate(preds).ravel()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Regression model — no predict_proba")

    def metric_history(self) -> list[dict]:
        rows = []
        for h in self.training_history_:
            epoch = h["epoch"]
            for split in ("train", "val"):
                key = f"{split}_mae"
                if key in h:
                    rows.append({"step": epoch, "split": split, "metrics": {"mae": h[key]}})
        return rows

    def _eval_mae(
        self, X_pep: np.ndarray, X_gen: np.ndarray, y: np.ndarray
    ) -> float:
        self._model.eval()
        with torch.no_grad():
            preds = []
            for start in range(0, len(X_pep), self.batch_size):
                pep_b = torch.tensor(X_pep[start:start + self.batch_size], device=self.device)
                gen_b = torch.tensor(X_gen[start:start + self.batch_size], device=self.device)
                preds.append(self._model(pep_b, gen_b).cpu().numpy())
            preds = np.concatenate(preds).ravel()
        return float(np.mean(np.abs(preds - y.ravel())))


# ─── Feature builder for registry ─────────────────────────────────────────────


def build_transformer_mic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build dual-branch features: peptide (ESM-2) + genome (oligo)."""
    genome_encoder = GenomeEncoder()

    # Peptide embeddings
    sequences = df["sequence"].astype(str).tolist()
    peptide_embs = _get_or_compute_esm2_embeddings(sequences)
    peptide_df = pd.DataFrame(
        peptide_embs,
        columns=[f"peptide_{i}" for i in range(peptide_embs.shape[1])],
        index=df.index,
    )

    # Genome embeddings
    target_names = df["target_activity_name"]
    genome_embs = genome_encoder.encode(target_names)
    genome_df = pd.DataFrame(
        genome_embs,
        columns=[f"genome_{i}" for i in range(genome_embs.shape[1])],
        index=df.index,
    )

    return pd.concat([peptide_df, genome_df], axis=1).astype(np.float32)
