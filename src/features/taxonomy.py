"""Taxonomy feature extraction for target species names."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

TAXONOMY_RANKS = ("phylum", "class", "order", "family", "genus")
TAXONOMY_COLUMNS = tuple(rank.capitalize() for rank in TAXONOMY_RANKS)
UNKNOWN = "Unknown"
BACTERIA_TAXID = 2

STRAIN_MARKERS = (
    "ATCC",
    "DSM",
    "MTCC",
    "NCTC",
    "JCM",
    "KCTC",
    "NBRC",
    "CIP",
    "CCUG",
    "CECT",
    "LMG",
    "PAO",
    "USA",
    "MRSA",
    "MSSA",
    "VRE",
    "VRSA",
    "strain",
    "str.",
    "isolate",
    "clinical isolate",
)

STRAIN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(marker) for marker in STRAIN_MARKERS) + r")\b.*",
    flags=re.IGNORECASE,
)
PUNCTUATION_PATTERN = re.compile(r"[\[\]\(\),;:]")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class TaxonomyLineage:
    """Normalized target-species taxonomy used as model features."""

    Phylum: str = UNKNOWN
    Class: str = UNKNOWN
    Order: str = UNKNOWN
    Family: str = UNKNOWN
    Genus: str = UNKNOWN
    is_bacteria: bool = False
    taxid: int | None = None
    query_name: str = ""
    matched_name: str = ""

    def as_feature_dict(self) -> dict[str, object]:
        return {
            "Phylum": self.Phylum,
            "Class": self.Class,
            "Order": self.Order,
            "Family": self.Family,
            "Genus": self.Genus,
            "is_bacteria": int(self.is_bacteria),
            "taxid": self.taxid,
            "taxonomy_query_name": self.query_name,
            "taxonomy_matched_name": self.matched_name,
        }


COMMON_TAXONOMY: dict[str, TaxonomyLineage] = {
    "Escherichia coli": TaxonomyLineage(
        Phylum="Pseudomonadota",
        Class="Gammaproteobacteria",
        Order="Enterobacterales",
        Family="Enterobacteriaceae",
        Genus="Escherichia",
        is_bacteria=True,
        query_name="Escherichia coli",
        matched_name="Escherichia coli",
    ),
    "Bacillus subtilis": TaxonomyLineage(
        Phylum="Bacillota",
        Class="Bacilli",
        Order="Bacillales",
        Family="Bacillaceae",
        Genus="Bacillus",
        is_bacteria=True,
        query_name="Bacillus subtilis",
        matched_name="Bacillus subtilis",
    ),
    "Staphylococcus aureus": TaxonomyLineage(
        Phylum="Bacillota",
        Class="Bacilli",
        Order="Bacillales",
        Family="Staphylococcaceae",
        Genus="Staphylococcus",
        is_bacteria=True,
        query_name="Staphylococcus aureus",
        matched_name="Staphylococcus aureus",
    ),
    "Pseudomonas aeruginosa": TaxonomyLineage(
        Phylum="Pseudomonadota",
        Class="Gammaproteobacteria",
        Order="Pseudomonadales",
        Family="Pseudomonadaceae",
        Genus="Pseudomonas",
        is_bacteria=True,
        query_name="Pseudomonas aeruginosa",
        matched_name="Pseudomonas aeruginosa",
    ),
}


def get_ncbi_taxa():
    """Create an ETE3 NCBI taxonomy adapter."""
    try:
        from ete3 import NCBITaxa
    except ImportError as exc:
        raise ImportError(
            "ete3 is required for taxonomy extraction. Install the project "
            "environment from env.yml or run `pip install ete3`."
        ) from exc
    return NCBITaxa()


def empty_lineage(query_name: str = "") -> TaxonomyLineage:
    return TaxonomyLineage(query_name=query_name)


def normalize_species_name(name: object) -> str:
    """Reduce DBAASP target names to a taxonomy-queryable organism name."""
    if pd.isna(name):
        return ""

    text = str(name).strip()
    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = STRAIN_PATTERN.sub("", text).strip()
    text = re.sub(r"\b(subsp\.?|ssp\.?|serovar|biovar|pv\.?|var\.?)\b.*", "", text, flags=re.IGNORECASE)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    words = text.split()
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    return text


def _lookup_taxids(ncbi, candidates: Iterable[str]) -> tuple[str, int] | None:
    for candidate in candidates:
        if not candidate:
            continue
        translated = ncbi.get_name_translator([candidate])
        if translated and candidate in translated and translated[candidate]:
            return candidate, int(translated[candidate][0])
    return None


def _lineage_from_taxid(ncbi, taxid: int, query_name: str, matched_name: str) -> TaxonomyLineage:
    lineage = ncbi.get_lineage(taxid)
    ranks = ncbi.get_rank(lineage)
    names = ncbi.get_taxid_translator(lineage)

    values = {column: UNKNOWN for column in TAXONOMY_COLUMNS}
    for rank_taxid, rank_name in ranks.items():
        if rank_name in TAXONOMY_RANKS:
            values[rank_name.capitalize()] = names.get(rank_taxid, UNKNOWN)

    is_bacteria = BACTERIA_TAXID in set(lineage)
    if not is_bacteria:
        superkingdom_taxids = [
            rank_taxid
            for rank_taxid, rank_name in ranks.items()
            if rank_name == "superkingdom"
        ]
        superkingdom_names = {names.get(rank_taxid) for rank_taxid in superkingdom_taxids}
        is_bacteria = "Bacteria" in superkingdom_names

    return TaxonomyLineage(
        **values,
        is_bacteria=is_bacteria,
        taxid=taxid,
        query_name=query_name,
        matched_name=matched_name,
    )


@lru_cache(maxsize=20000)
def get_taxonomic_lineage(name: object, ncbi=None) -> TaxonomyLineage:
    """Return bacterial taxonomy ranks for one target organism name."""
    query_name = normalize_species_name(name)
    if not query_name:
        return empty_lineage()

    if query_name in COMMON_TAXONOMY:
        return COMMON_TAXONOMY[query_name]

    if ncbi is None:
        ncbi = get_ncbi_taxa()

    candidates = [query_name, str(name).strip()]
    try:
        lookup = _lookup_taxids(ncbi, candidates)
        if lookup is None:
            return empty_lineage(query_name=query_name)
        matched_name, taxid = lookup
        lineage = _lineage_from_taxid(ncbi, taxid, query_name, matched_name)
        if not lineage.is_bacteria:
            return empty_lineage(query_name=query_name)
        return lineage
    except Exception:
        return empty_lineage(query_name=query_name)


def build_taxonomy_lookup(
    target_names: Iterable[object],
    ncbi=None,
) -> pd.DataFrame:
    """Build one row of taxonomy ranks per unique target name."""
    unique_names = pd.Series(list(target_names), name="target_activity_name").dropna().drop_duplicates()
    rows = []
    for target_name in unique_names:
        lineage = get_taxonomic_lineage(target_name, ncbi=ncbi).as_feature_dict()
        lineage["target_activity_name"] = target_name
        rows.append(lineage)
    return pd.DataFrame(rows)


def add_taxonomy_columns(
    df: pd.DataFrame,
    target_col: str = "target_activity_name",
    ncbi=None,
) -> pd.DataFrame:
    """Merge taxonomy rank columns onto an activity dataframe."""
    if target_col not in df.columns:
        raise ValueError(f"Missing required target column: {target_col}")
    lookup = build_taxonomy_lookup(df[target_col], ncbi=ncbi)
    return df.merge(lookup, on=target_col, how="left")


def build_taxonomy_feature_matrix(
    df: pd.DataFrame,
    rank_columns: Iterable[str] = TAXONOMY_COLUMNS,
    include_is_bacteria: bool = True,
) -> pd.DataFrame:
    """Convert taxonomy ranks to fixed-width one-hot model features."""
    ranks = list(rank_columns)
    missing = [column for column in ranks if column not in df.columns]
    if missing:
        raise ValueError(f"Missing taxonomy columns: {missing}")

    rank_df = df[ranks].fillna(UNKNOWN)
    feature_df = pd.get_dummies(rank_df, prefix=ranks, dtype=int)
    if include_is_bacteria and "is_bacteria" in df.columns:
        feature_df["target_is_bacteria"] = df["is_bacteria"].fillna(0).astype(int)
    return feature_df


def write_taxonomy_feature_file(
    input_csv: str | Path,
    output_csv: str | Path,
    target_col: str = "target_activity_name",
) -> pd.DataFrame:
    """Load activities, append taxonomy columns and one-hot features, then save."""
    df = pd.read_csv(input_csv)
    enriched = add_taxonomy_columns(df, target_col=target_col)
    feature_matrix = build_taxonomy_feature_matrix(enriched)
    output = pd.concat([enriched, feature_matrix], axis=1)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output
