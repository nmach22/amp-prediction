"""
Compute ESM-2 embeddings for sequences in a raw CSV file.

Unlike make_plm_embeddings.py, this only requires a 'sequence' column —
no gram_status or taxonomy needed. Shows progress and saves incrementally.

Usage:
    python scripts/compute_esm2_embeddings.py \
        --input data/raw/amp_mic_activities.csv \
        --device mps --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.plm import (
    DEFAULT_ESM2_MODEL,
    DEFAULT_MIC_EMBEDDING_PATH,
    PLMEncoder,
    load_embedding_cache,
    save_embedding_cache,
)


def main():
    parser = argparse.ArgumentParser(description="Compute ESM-2 embeddings")
    parser.add_argument("--input", required=True, help="CSV with 'sequence' column")
    parser.add_argument("--output", default=str(DEFAULT_MIC_EMBEDDING_PATH))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # Load sequences
    df = pd.read_csv(args.input)
    all_seqs = sorted(df["sequence"].dropna().str.upper().str.strip().unique().tolist())
    # Filter non-standard amino acids
    import re
    all_seqs = [s for s in all_seqs if s and not re.search(r"[^ACDEFGHIKLMNPQRSTVWY]", s)]
    print(f"Total unique valid sequences: {len(all_seqs)}")

    # Load existing cache
    output_path = Path(args.output)
    cached_seqs, cached_embs = [], np.empty((0, 0))
    cache_index = {}
    if output_path.exists():
        cached_seqs, cached_embs = load_embedding_cache(output_path)
        cache_index = set(cached_seqs)
        print(f"Existing cache: {len(cached_seqs)} sequences ({cached_embs.shape[1]}d)")

    # Find missing
    missing = [s for s in all_seqs if s not in cache_index]
    print(f"Missing sequences: {len(missing)}")

    if not missing:
        print("All sequences already cached. Nothing to do.")
        return

    # Compute missing embeddings in batches with progress
    print(f"\nComputing ESM-2 embeddings on {args.device} (batch_size={args.batch_size})...")
    encoder = PLMEncoder(model_name=DEFAULT_ESM2_MODEL, cache_dir=None, device=args.device)

    # Process in chunks and save incrementally
    chunk_size = 200  # save every 200 sequences
    new_seqs = []
    new_embs = []
    t0 = time.time()

    for chunk_start in range(0, len(missing), chunk_size):
        chunk = missing[chunk_start:chunk_start + chunk_size]
        embeddings = encoder.encode(chunk, batch_size=args.batch_size)
        new_seqs.extend(chunk)
        new_embs.append(embeddings)

        done = chunk_start + len(chunk)
        elapsed = time.time() - t0
        rate = done / elapsed
        remaining = (len(missing) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(missing)}] {rate:.1f} seq/s, ~{remaining/60:.1f} min remaining",
              flush=True)

        # Incremental save
        all_new_embs = np.vstack(new_embs)
        if len(cached_embs) > 0 and cached_embs.shape[1] > 0:
            merged_seqs = list(cached_seqs) + new_seqs
            merged_embs = np.vstack([cached_embs, all_new_embs])
        else:
            merged_seqs = new_seqs
            merged_embs = all_new_embs

        save_embedding_cache(
            output_path, merged_seqs, merged_embs, model_name=DEFAULT_ESM2_MODEL
        )

    total_time = time.time() - t0
    print(f"\nDone! Computed {len(missing)} embeddings in {total_time/60:.1f} min")
    print(f"Total cache: {len(merged_seqs)} sequences → {output_path}")


if __name__ == "__main__":
    main()
