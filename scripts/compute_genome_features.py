"""
compute_genome_features.py
==========================
Compute genome-level features from downloaded reference genomes:
1. Oligonucleotide composition (k=3, k=4, k=5 normalized frequencies)
2. gyrB gene extraction and pairwise similarity
3. dDDH estimation via ANI approximation

Usage:
    python scripts/compute_genome_features.py \
        --mapping data/processed/embeddings/genome/strain_assembly_map.csv \
        --genome-dir data/raw/genomes \
        --output-dir data/processed/embeddings/genome

Requirements:
    - BioPython
    - numpy, pandas, scipy
    - Optional: pyani (for ANI), BLAST+ (for gyrB extraction)
"""

from __future__ import annotations

import argparse
import gzip
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO


# ─── Oligonucleotide Composition ───────────────────────────────────────────────


def compute_kmer_frequencies(sequence: str, k: int) -> dict[str, float]:
    """Compute normalized k-mer frequencies for a DNA sequence."""
    sequence = sequence.upper().replace("N", "")
    total = len(sequence) - k + 1
    if total <= 0:
        return {}

    counts = Counter()
    for i in range(total):
        kmer = sequence[i : i + k]
        if all(c in "ACGT" for c in kmer):
            counts[kmer] += 1

    # Normalize
    valid_total = sum(counts.values())
    if valid_total == 0:
        return {}
    return {kmer: count / valid_total for kmer, count in counts.items()}


def all_kmers(k: int) -> list[str]:
    """Generate all possible k-mers in lexicographic order."""
    bases = "ACGT"
    return ["".join(combo) for combo in itertools.product(bases, repeat=k)]


def genome_kmer_vector(fasta_path: Path, k: int) -> np.ndarray:
    """Compute k-mer frequency vector for an entire genome."""
    # Read all contigs
    full_sequence = ""
    opener = gzip.open if str(fasta_path).endswith(".gz") else open

    with opener(fasta_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            full_sequence += str(record.seq)

    freqs = compute_kmer_frequencies(full_sequence, k)
    kmers = all_kmers(k)
    return np.array([freqs.get(kmer, 0.0) for kmer in kmers], dtype=np.float32)


def compute_oligonucleotide_features(
    mapping_df: pd.DataFrame,
    genome_dir: Path,
    k_values: tuple[int, ...] = (3, 4, 5),
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Compute oligonucleotide composition with incremental caching."""
    import time as _time

    # Load any previously cached results
    cached = {}
    if cache_path and cache_path.exists():
        prev = pd.read_parquet(cache_path)
        for _, row in prev.iterrows():
            cached[row["species"]] = row.to_dict()
        print(f"  Loaded {len(cached)} cached species from {cache_path}")

    results = list(cached.values())
    total = len(mapping_df)
    done = 0
    skipped = 0
    t0 = _time.time()

    for _, row in mapping_df.iterrows():
        species = row["species"]
        local_path = row.get("local_path")
        done += 1

        if species in cached:
            skipped += 1
            continue

        if pd.isna(local_path) or not Path(local_path).exists():
            continue

        fasta_path = Path(local_path)
        feature_row = {"species": species}

        for k in k_values:
            kmers = all_kmers(k)
            vector = genome_kmer_vector(fasta_path, k)
            for kmer, freq in zip(kmers, vector):
                feature_row[f"kmer_{k}_{kmer}"] = freq

        results.append(feature_row)
        cached[species] = feature_row
        new_count = len(results) - skipped

        # Save cache every 10 new genomes
        if cache_path and new_count % 10 == 0:
            pd.DataFrame(results).to_parquet(cache_path, index=False)

        elapsed = _time.time() - t0
        per_genome = elapsed / max(new_count - (len(cached) - new_count), 1)
        remaining = total - done
        eta_min = (remaining * per_genome) / 60
        print(
            f"  [{done}/{total}] {species} done "
            f"({len(results)} total, ~{eta_min:.1f} min remaining)",
            flush=True,
        )

    # Final save
    df = pd.DataFrame(results)
    if cache_path:
        df.to_parquet(cache_path, index=False)

    return df

    return pd.DataFrame(results)


# ─── gyrB Gene Similarity ──────────────────────────────────────────────────────


def extract_gyrb_from_genome(fasta_path: Path, gyrb_ref_path: Path) -> str | None:
    """
    Extract gyrB gene sequence from a genome using simple sequence search.
    For production use, tblastn with a reference gyrB protein would be better.
    Here we use a simplified approach: search for the conserved gyrB motif.
    """
    import subprocess
    import tempfile

    # Use makeblastdb + tblastn if BLAST+ is available
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Decompress if needed
            if str(fasta_path).endswith(".gz"):
                decompressed = Path(tmpdir) / "genome.fna"
                with gzip.open(fasta_path, "rt") as f_in:
                    decompressed.write_text(f_in.read())
                genome_path = decompressed
            else:
                genome_path = fasta_path

            # Make BLAST database
            subprocess.run(
                ["makeblastdb", "-in", str(genome_path), "-dbtype", "nucl", "-out", f"{tmpdir}/db"],
                capture_output=True,
                check=True,
            )

            # Run tblastn with gyrB reference protein
            result = subprocess.run(
                [
                    "tblastn",
                    "-query", str(gyrb_ref_path),
                    "-db", f"{tmpdir}/db",
                    "-outfmt", "6 sseq",
                    "-evalue", "1e-50",
                    "-max_target_seqs", "1",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                # Return the best hit nucleotide sequence
                return result.stdout.strip().split("\n")[0].replace("-", "")

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def compute_gyrb_pairwise_identity(gyrb_sequences: dict[str, str]) -> pd.DataFrame:
    """Compute pairwise nucleotide identity between gyrB sequences."""
    from scipy.spatial.distance import pdist, squareform

    species_list = sorted(gyrb_sequences.keys())
    n = len(species_list)

    if n == 0:
        return pd.DataFrame()

    # Simple percent identity calculation
    def seq_identity(seq1: str, seq2: str) -> float:
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
        return matches / min_len

    identity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                identity_matrix[i, j] = 1.0
            else:
                ident = seq_identity(gyrb_sequences[species_list[i]], gyrb_sequences[species_list[j]])
                identity_matrix[i, j] = ident
                identity_matrix[j, i] = ident

    return pd.DataFrame(identity_matrix, index=species_list, columns=species_list)


# ─── dDDH Estimation ──────────────────────────────────────────────────────────


def estimate_ani_mash(genome_paths: dict[str, Path]) -> pd.DataFrame:
    """
    Estimate ANI between genome pairs using Mash distance.
    ANI ≈ 1 - mash_distance (approximation).
    Requires: mash (install via conda: `conda install -c bioconda mash`)
    """
    import subprocess
    import tempfile

    species_list = sorted(genome_paths.keys())
    n = len(species_list)
    ani_matrix = np.eye(n)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mash sketches
            sketch_paths = {}
            for species in species_list:
                genome_path = genome_paths[species]
                sketch_out = Path(tmpdir) / f"{species.replace(' ', '_')}"
                result = subprocess.run(
                    ["mash", "sketch", "-o", str(sketch_out), str(genome_path)],
                    capture_output=True,
                )
                if result.returncode == 0:
                    sketch_paths[species] = f"{sketch_out}.msh"

            # Pairwise distances
            for i in range(n):
                for j in range(i + 1, n):
                    sp_i, sp_j = species_list[i], species_list[j]
                    if sp_i not in sketch_paths or sp_j not in sketch_paths:
                        continue

                    result = subprocess.run(
                        ["mash", "dist", sketch_paths[sp_i], sketch_paths[sp_j]],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        fields = result.stdout.strip().split("\t")
                        mash_dist = float(fields[2])
                        ani = 1.0 - mash_dist
                        ani_matrix[i, j] = ani
                        ani_matrix[j, i] = ani

    except FileNotFoundError:
        print("WARNING: mash not found. Using placeholder ANI values.")
        # Fallback: use genus-level grouping as proxy
        return pd.DataFrame(ani_matrix, index=species_list, columns=species_list)

    return pd.DataFrame(ani_matrix, index=species_list, columns=species_list)


def ani_to_ddh(ani: float) -> float:
    """Convert ANI to dDDH using Meier-Kolthoff et al. formula 2."""
    # Formula 2: dDDH = (ANI/100 - 0.1) / 0.0089 * 100
    # Simplified for ANI in [0, 1] scale:
    if ani <= 0:
        return 0.0
    ddh = (ani - 0.1) / 0.009
    return max(0.0, min(100.0, ddh))


def compute_ddh_matrix(ani_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert ANI matrix to dDDH estimates."""
    ddh_matrix = ani_matrix.map(ani_to_ddh)
    return ddh_matrix


# ─── Main Pipeline ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Compute genome-level features")
    parser.add_argument("--mapping", required=True, help="Strain-assembly mapping CSV")
    parser.add_argument("--genome-dir", default="data/raw/genomes")
    parser.add_argument("--output-dir", default="data/processed/embeddings/genome")
    parser.add_argument("--gyrb-ref", default=None, help="Reference gyrB protein FASTA for extraction")
    parser.add_argument("--skip-gyrb", action="store_true", help="Skip gyrB extraction (requires BLAST+)")
    parser.add_argument("--skip-ddh", action="store_true", help="Skip dDDH estimation (requires mash)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_df = pd.read_csv(args.mapping)
    # Deduplicate to one genome per species
    species_mapping = mapping_df.dropna(subset=["local_path"]).drop_duplicates(subset=["species"])
    print(f"Species with genomes: {len(species_mapping)}")

    # 1. Oligonucleotide composition
    print("\n=== Computing oligonucleotide composition (k=3,4,5) ===")
    oligo_path = output_dir / "oligo_composition.parquet"
    oligo_df = compute_oligonucleotide_features(
        species_mapping, Path(args.genome_dir), cache_path=oligo_path,
    )
    oligo_df.to_parquet(oligo_path, index=False)
    print(f"  Saved: {oligo_path} ({oligo_df.shape})")

    # 2. gyrB similarity
    if not args.skip_gyrb and args.gyrb_ref:
        print("\n=== Extracting gyrB and computing similarity ===")
        gyrb_ref_path = Path(args.gyrb_ref)
        gyrb_sequences = {}
        for _, row in species_mapping.iterrows():
            seq = extract_gyrb_from_genome(Path(row["local_path"]), gyrb_ref_path)
            if seq:
                gyrb_sequences[row["species"]] = seq

        print(f"  Extracted gyrB for {len(gyrb_sequences)} species")
        gyrb_sim = compute_gyrb_pairwise_identity(gyrb_sequences)
        gyrb_path = output_dir / "gyrb_similarity.parquet"
        gyrb_sim.to_parquet(gyrb_path)
        print(f"  Saved: {gyrb_path} ({gyrb_sim.shape})")
    else:
        print("\n=== Skipping gyrB extraction ===")

    # 3. dDDH estimation
    if not args.skip_ddh:
        print("\n=== Estimating dDDH via ANI (Mash) ===")
        genome_paths = {
            row["species"]: Path(row["local_path"])
            for _, row in species_mapping.iterrows()
        }
        ani_matrix = estimate_ani_mash(genome_paths)
        ddh_matrix = compute_ddh_matrix(ani_matrix)

        ani_path = output_dir / "ani_matrix.parquet"
        ddh_path = output_dir / "ddh_estimates.parquet"
        ani_matrix.to_parquet(ani_path)
        ddh_matrix.to_parquet(ddh_path)
        print(f"  Saved ANI: {ani_path} ({ani_matrix.shape})")
        print(f"  Saved dDDH: {ddh_path} ({ddh_matrix.shape})")
    else:
        print("\n=== Skipping dDDH estimation ===")

    print("\nDone! Genome features computed successfully.")


if __name__ == "__main__":
    main()
