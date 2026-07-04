"""Precompute frozen PLM embeddings for MIC regression peptide sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.plm import (  # noqa: E402
    DEFAULT_ESM2_MODEL,
    DEFAULT_MIC_EMBEDDING_PATH,
    PLMEncoder,
    save_embedding_cache,
)
from src.models.xgboost_mic import load_xgboost_mic_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute frozen ESM2 embeddings for unique MIC peptides."
    )
    parser.add_argument(
        "--input",
        default="data/processed/amp_mic_activities_taxonomy_features.csv",
        help="MIC taxonomy CSV used by the MIC regression experiments.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MIC_EMBEDDING_PATH),
        help="Output .npz cache path.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_ESM2_MODEL,
        help="HuggingFace protein language model identifier.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Torch device used for embedding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Sequences per embedding batch.",
    )
    return parser.parse_args()


def build_embedding_cache(
    *,
    input_csv: str | Path,
    output_path: str | Path,
    model_name: str = DEFAULT_ESM2_MODEL,
    device: str = "cpu",
    batch_size: int = 16,
) -> Path:
    """Compute embeddings for each unique cleaned MIC peptide sequence."""
    input_csv = Path(input_csv)
    output_path = Path(output_path)
    df = load_xgboost_mic_data(input_csv)
    sequences = sorted(df["sequence"].astype(str).unique().tolist())
    if not sequences:
        raise ValueError(f"No valid peptide sequences found in {input_csv}.")

    encoder = PLMEncoder(model_name=model_name, cache_dir=None, device=device)
    embeddings = encoder.encode(sequences, batch_size=batch_size)
    return save_embedding_cache(
        output_path,
        sequences,
        embeddings,
        model_name=model_name,
        metadata={
            "source_path": str(input_csv),
            "device": device,
            "batch_size": int(batch_size),
            "source_row_count": int(len(df)),
        },
    )


def main() -> None:
    args = parse_args()
    path = build_embedding_cache(
        input_csv=args.input,
        output_path=args.output,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"Saved PLM embeddings to {path}")


if __name__ == "__main__":
    main()
