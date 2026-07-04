from pathlib import Path

import numpy as np
import pandas as pd

from scripts import make_plm_embeddings
from src.features.plm import (
    embeddings_for_sequences,
    load_embedding_cache_metadata,
    save_embedding_cache,
)


def test_embedding_cache_loads_rows_in_requested_order(tmp_path):
    path = tmp_path / "cache.npz"
    save_embedding_cache(
        path,
        ["AAAA", "CCCC"],
        np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32),
        model_name="facebook/esm2_t12_35M_UR50D",
    )

    embeddings = embeddings_for_sequences(["CCCC", "AAAA", "CCCC"], path)

    assert embeddings.tolist() == [[2.0, 2.5], [1.0, 1.5], [2.0, 2.5]]


def test_make_plm_embeddings_embeds_unique_sequences_once(tmp_path, monkeypatch):
    input_csv = tmp_path / "mic.csv"
    output_path = tmp_path / "embeddings.npz"
    pd.DataFrame({"sequence": ["CCCC", "AAAA", "CCCC"]}).to_csv(
        input_csv,
        index=False,
    )

    monkeypatch.setattr(
        make_plm_embeddings,
        "load_xgboost_mic_data",
        lambda path: pd.DataFrame({"sequence": ["CCCC", "AAAA", "CCCC"]}),
    )

    class FakePLMEncoder:
        def __init__(self, model_name: str, cache_dir: Path | None, device: str):
            self.model_name = model_name
            self.cache_dir = cache_dir
            self.device = device

        def encode(self, sequences: list[str], batch_size: int = 16) -> np.ndarray:
            assert sequences == ["AAAA", "CCCC"]
            assert batch_size == 4
            return np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)

    monkeypatch.setattr(make_plm_embeddings, "PLMEncoder", FakePLMEncoder)

    path = make_plm_embeddings.build_embedding_cache(
        input_csv=input_csv,
        output_path=output_path,
        model_name="fake/esm2",
        device="cpu",
        batch_size=4,
    )

    assert path == output_path
    assert embeddings_for_sequences(["CCCC", "AAAA"], output_path).tolist() == [
        [2.0, 2.5],
        [1.0, 1.5],
    ]
    metadata = load_embedding_cache_metadata(output_path)
    assert metadata["model_name"] == "fake/esm2"
    assert metadata["sequence_count"] == 2
    assert metadata["source_path"] == str(input_csv)
