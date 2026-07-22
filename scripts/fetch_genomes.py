"""
fetch_genomes.py
================
Map dataset target organisms to NCBI genome assemblies and download reference
genomes at strain level (falling back to species type-strain).

Usage:
    python scripts/fetch_genomes.py \
        --input data/processed/splits/train.csv \
        --output-dir data/raw/genomes \
        --mapping-out data/processed/embeddings/genome/strain_assembly_map.csv

Requirements:
    - BioPython (`pip install biopython`)
    - Internet access for NCBI Entrez queries

The script:
1. Extracts unique target organisms from the dataset.
2. Queries NCBI Assembly DB via Entrez for each species/strain.
3. Downloads the best matching genome assembly (.fna.gz).
4. Produces a mapping CSV: target_activity_name → species → assembly_accession → local_path
"""

from __future__ import annotations

import argparse
import gzip
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from Bio import Entrez

# Must set email for NCBI Entrez
Entrez.email = "amp.prediction.pipeline@example.com"
Entrez.api_key = None  # Set via env var NCBI_API_KEY if available

STRAIN_MARKERS = re.compile(
    r"\b(ATCC|DSM|MTCC|NCTC|JCM|KCTC|NBRC|CIP|CCUG|CECT|LMG)\s*[\-]?\s*(\d+\w*)",
    re.IGNORECASE,
)


def extract_species(name: str) -> str:
    """Extract binomial species name (first two words)."""
    words = str(name).strip().split()
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    return name


def extract_strain_id(name: str) -> Optional[str]:
    """Extract culture collection ID if present (e.g., 'ATCC 6538P')."""
    match = STRAIN_MARKERS.search(str(name))
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None


def search_assembly(organism: str, strain_id: Optional[str] = None) -> Optional[dict]:
    """
    Search NCBI Assembly DB for a genome matching the organism.
    Prefer RefSeq, complete genomes, strain-level matches.
    """
    # Build search query
    query_parts = [f'"{organism}"[Organism]']
    if strain_id:
        query_parts.append(f'"{strain_id}"')
    query_parts.append('"latest refseq"[filter]')

    query = " AND ".join(query_parts)

    try:
        handle = Entrez.esearch(db="assembly", term=query, retmax=5)
        record = Entrez.read(handle)
        handle.close()

        if not record["IdList"]:
            # Fallback: search without strain, just species
            query = f'"{organism}"[Organism] AND "latest refseq"[filter] AND "representative genome"[filter]'
            handle = Entrez.esearch(db="assembly", term=query, retmax=3)
            record = Entrez.read(handle)
            handle.close()

        if not record["IdList"]:
            # Last fallback: any genome for this species
            query = f'"{organism}"[Organism] AND "latest refseq"[filter]'
            handle = Entrez.esearch(db="assembly", term=query, retmax=1)
            record = Entrez.read(handle)
            handle.close()

        if not record["IdList"]:
            return None

        # Get assembly details
        assembly_id = record["IdList"][0]
        handle = Entrez.esummary(db="assembly", id=assembly_id, report="full")
        summary = Entrez.read(handle)
        handle.close()

        doc = summary["DocumentSummarySet"]["DocumentSummary"][0]
        accession = doc.get("AssemblyAccession", "")
        ftp_path = doc.get("FtpPath_RefSeq", "") or doc.get("FtpPath_GenBank", "")
        organism_name = doc.get("Organism", "")
        assembly_name = doc.get("AssemblyName", "")

        return {
            "assembly_accession": accession,
            "ftp_path": ftp_path,
            "organism_name": organism_name,
            "assembly_name": assembly_name,
            "assembly_id": assembly_id,
        }

    except Exception as e:
        print(f"  ERROR querying '{organism}': {e}")
        return None


def download_genome(ftp_path: str, output_dir: Path, accession: str) -> Optional[Path]:
    """Download genome FASTA from NCBI FTP with streaming and timeout."""
    import shutil
    import ssl
    import urllib.request

    if not ftp_path:
        return None

    # Construct the genomic fasta URL
    asm_name = ftp_path.split("/")[-1]
    url = f"{ftp_path}/{asm_name}_genomic.fna.gz"

    # Convert ftp:// to https:// for compatibility
    url = url.replace("ftp://", "https://")

    output_path = output_dir / f"{accession}_genomic.fna.gz"
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    # macOS Python often lacks the NCBI FTP CA in its cert bundle;
    # fall back to unverified SSL for this public data source.
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    tmp_path = output_path.with_suffix(".tmp")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as response, \
             open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file, length=1024 * 64)
        tmp_path.rename(output_path)
        return output_path
    except Exception as e:
        print(f"    ERROR {accession}: {e}")
        for p in (tmp_path, output_path):
            if p.exists():
                p.unlink()
        return None


def build_strain_mapping(input_csv: str) -> pd.DataFrame:
    """Build a dataframe of unique strains with species and strain IDs."""
    df = pd.read_csv(input_csv)
    targets = df["target_activity_name"].unique()

    rows = []
    for target in sorted(targets):
        species = extract_species(target)
        strain_id = extract_strain_id(target)
        rows.append({
            "target_activity_name": target,
            "species": species,
            "strain_id": strain_id,
        })

    return pd.DataFrame(rows)


def _save_search_cache(assembly_map: dict, path: Path) -> None:
    """Persist NCBI search results so re-runs skip already-found species."""
    rows = []
    for species, info in assembly_map.items():
        rows.append({"species": species, **info})
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Fetch genomes for dataset organisms")
    parser.add_argument("--input", required=True, help="Input CSV with target_activity_name column")
    parser.add_argument("--output-dir", default="data/raw/genomes", help="Directory for genome downloads")
    parser.add_argument("--mapping-out", default="data/processed/embeddings/genome/strain_assembly_map.csv")
    parser.add_argument("--max-species", type=int, default=None, help="Limit species to process (for testing)")
    parser.add_argument("--delay", type=float, default=0.4, help="Delay between NCBI queries (seconds)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = Path(args.mapping_out)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    # Intermediate cache for NCBI search results (survives restarts)
    search_cache_path = mapping_path.parent / "assembly_search_cache.csv"

    # Build strain mapping
    strain_df = build_strain_mapping(args.input)
    print(f"Total unique targets: {len(strain_df)}")
    print(f"Unique species: {strain_df['species'].nunique()}")

    # Process species-level (one genome per species, best match)
    species_list = strain_df["species"].unique()
    if args.max_species:
        species_list = species_list[: args.max_species]

    # Load cached search results if available
    assembly_map = {}
    if search_cache_path.exists():
        cache_df = pd.read_csv(search_cache_path)
        for _, row in cache_df.iterrows():
            assembly_map[row["species"]] = {
                "assembly_accession": row["assembly_accession"],
                "ftp_path": row["ftp_path"],
                "organism_name": row["organism_name"],
                "assembly_name": row.get("assembly_name", ""),
                "assembly_id": row.get("assembly_id", ""),
            }
        print(f"\nLoaded {len(assembly_map)} cached search results from {search_cache_path}")

    # Only search species we haven't cached yet
    species_to_search = [s for s in species_list if s not in assembly_map]
    if species_to_search:
        print(f"\nSearching NCBI for {len(species_to_search)} species "
              f"({len(species_list) - len(species_to_search)} already cached)...")

        for i, species in enumerate(species_to_search):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(species_to_search)}")

            species_strains = strain_df[strain_df["species"] == species]
            strain_ids = species_strains["strain_id"].dropna().unique()
            best_strain_id = strain_ids[0] if len(strain_ids) > 0 else None

            result = search_assembly(species, strain_id=best_strain_id)
            if result:
                assembly_map[species] = result

            # Save cache after every 20 species so progress is never lost
            if (i + 1) % 20 == 0 or i == len(species_to_search) - 1:
                _save_search_cache(assembly_map, search_cache_path)

            time.sleep(args.delay)

        print(f"\nFound assemblies for {len(assembly_map)}/{len(species_list)} species")
    else:
        print(f"\nAll {len(species_list)} species already cached, skipping search.")

    # Download genomes
    total = len(assembly_map)
    print(f"\nDownloading {total} genomes...")
    download_results = {}
    for i, (species, info) in enumerate(assembly_map.items(), 1):
        accession = info["assembly_accession"]
        output_path = output_dir / f"{accession}_genomic.fna.gz"
        already = output_path.exists() and output_path.stat().st_size > 0
        status = "cached" if already else "downloading"
        print(f"  [{i}/{total}] {species} ({accession}) ... {status}", flush=True)
        local_path = download_genome(info["ftp_path"], output_dir, accession)
        download_results[species] = {
            **info,
            "local_path": str(local_path) if local_path else None,
        }
        if not already:
            time.sleep(args.delay)

    # Build final mapping: target_activity_name → species → assembly → local_path
    mapping_rows = []
    for _, row in strain_df.iterrows():
        species = row["species"]
        if species in download_results:
            info = download_results[species]
            mapping_rows.append({
                "target_activity_name": row["target_activity_name"],
                "species": species,
                "strain_id": row["strain_id"],
                "assembly_accession": info["assembly_accession"],
                "organism_name": info["organism_name"],
                "local_path": info["local_path"],
            })
        else:
            mapping_rows.append({
                "target_activity_name": row["target_activity_name"],
                "species": species,
                "strain_id": row["strain_id"],
                "assembly_accession": None,
                "organism_name": None,
                "local_path": None,
            })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(mapping_path, index=False)
    print(f"\nMapping saved to {mapping_path}")
    print(f"  Total targets: {len(mapping_df)}")
    print(f"  With genome: {mapping_df['local_path'].notna().sum()}")
    print(f"  Missing genome: {mapping_df['local_path'].isna().sum()}")


if __name__ == "__main__":
    main()
