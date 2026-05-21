import pandas as pd

from src.features.taxonomy import (
    BACTERIA_TAXID,
    add_taxonomy_columns,
    build_taxonomy_feature_matrix,
    get_taxonomic_lineage,
    normalize_species_name,
)


class FakeNCBITaxa:
    def __init__(self):
        self.name_to_taxid = {
            "Bacillus subtilis": [1423],
            "Listeria monocytogenes": [1639],
            "Candida albicans": [5476],
        }
        self.lineages = {
            1423: [1, BACTERIA_TAXID, 1239, 91061, 1385, 186817, 1386, 1423],
            1639: [1, BACTERIA_TAXID, 1239, 91061, 1385, 186820, 1637, 1639],
            5476: [1, 2759, 33154, 4751, 4890, 147537, 5476],
        }
        self.ranks = {
            1: "no rank",
            BACTERIA_TAXID: "superkingdom",
            1239: "phylum",
            91061: "class",
            1385: "order",
            186817: "family",
            1386: "genus",
            1423: "species",
            186820: "family",
            1637: "genus",
            1639: "species",
            2759: "superkingdom",
            33154: "kingdom",
            4751: "phylum",
            4890: "class",
            147537: "genus",
            5476: "species",
        }
        self.names = {
            1: "root",
            BACTERIA_TAXID: "Bacteria",
            1239: "Bacillota",
            91061: "Bacilli",
            1385: "Bacillales",
            186817: "Bacillaceae",
            1386: "Bacillus",
            1423: "Bacillus subtilis",
            186820: "Listeriaceae",
            1637: "Listeria",
            1639: "Listeria monocytogenes",
            2759: "Eukaryota",
            33154: "Opisthokonta",
            4751: "Ascomycota",
            4890: "Saccharomycetes",
            147537: "Candida",
            5476: "Candida albicans",
        }

    def get_name_translator(self, names):
        return {
            name: self.name_to_taxid[name]
            for name in names
            if name in self.name_to_taxid
        }

    def get_lineage(self, taxid):
        return self.lineages[taxid]

    def get_rank(self, lineage):
        return {taxid: self.ranks[taxid] for taxid in lineage}

    def get_taxid_translator(self, lineage):
        return {taxid: self.names[taxid] for taxid in lineage}


def test_normalize_species_name_removes_strain_suffixes():
    assert (
        normalize_species_name("Staphylococcus aureus ATCC 25923")
        == "Staphylococcus aureus"
    )
    assert normalize_species_name("Bacillus subtilis strain PY22") == "Bacillus subtilis"
    assert (
        normalize_species_name("Salmonella enterica subsp. enterica serovar Typhimurium")
        == "Salmonella enterica"
    )


def test_get_taxonomic_lineage_maps_ete3_lowercase_ranks():
    lineage = get_taxonomic_lineage("Listeria monocytogenes EGD-e", ncbi=FakeNCBITaxa())

    assert lineage.is_bacteria
    assert lineage.Phylum == "Bacillota"
    assert lineage.Class == "Bacilli"
    assert lineage.Order == "Bacillales"
    assert lineage.Family == "Listeriaceae"
    assert lineage.Genus == "Listeria"


def test_get_taxonomic_lineage_filters_non_bacteria():
    lineage = get_taxonomic_lineage("Candida albicans ATCC 10231", ncbi=FakeNCBITaxa())

    assert not lineage.is_bacteria
    assert lineage.Phylum == "Unknown"
    assert lineage.Genus == "Unknown"


def test_add_taxonomy_columns_and_one_hot_features():
    df = pd.DataFrame(
        {
            "sequence": ["AAAA", "CCCC"],
            "target_activity_name": [
                "Bacillus subtilis PY22",
                "Candida albicans ATCC 10231",
            ],
            "activity": [1.0, 2.0],
        }
    )

    enriched = add_taxonomy_columns(df, ncbi=FakeNCBITaxa())
    features = build_taxonomy_feature_matrix(enriched)

    assert "Phylum" in enriched.columns
    assert "Genus_Bacillus" in features.columns
    assert "Genus_Unknown" in features.columns
    assert "target_is_bacteria" in features.columns
    assert features["target_is_bacteria"].tolist() == [1, 0]
