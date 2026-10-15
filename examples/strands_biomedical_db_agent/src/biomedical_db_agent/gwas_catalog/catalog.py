"""SQL catalog over unmodified official GWAS Parquet files; no agent tools."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import duckdb

RELEASE = "2026-03-17"
SCHEMA_VERSION = 1
FILE_HEADERS_URL = "https://www.ebi.ac.uk/gwas/docs/fileheaders"
DATA_ROOT_ENV = "GWAS_CATALOG_SQL_DATA_ROOT"
DEFAULT_DATA_ROOT = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "gwas_catalog" / RELEASE


@dataclass(frozen=True)
class Column:
    source: str
    name: str
    type: str
    description: str

    @property
    def comment(self) -> str:
        return f"Source: {self.source}. {self.description}"


@dataclass(frozen=True)
class Dataset:
    name: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int
    columns: tuple[Column, ...]
    description: str

    @property
    def source_url(self) -> str:
        return f"https://huggingface.co/datasets/gwascatalog/{self.name}/resolve/{self.revision}/{self.filename}"


_FIELDS = [
    Column("DATE ADDED TO CATALOG", "date_added_to_catalog", "DATE", "Date published in the Catalog."),
    Column("PUBMEDID", "pubmed_id", "BIGINT", "PubMed publication identifier; several studies can share it."),
    Column("FIRST AUTHOR", "first_author", "VARCHAR", "First author as supplied."),
    Column("DATE", "publication_date", "DATE", "Publication date, not the Catalog release date."),
    Column("JOURNAL", "journal", "VARCHAR", "Journal name as supplied."),
    Column("LINK", "publication_link", "VARCHAR", "Publication link as supplied; a URL scheme may be absent."),
    Column("STUDY", "study_title", "VARCHAR", "Paper title; accession_id identifies the Catalog study."),
    Column("DISEASE/TRAIT", "disease_trait", "VARCHAR", "Reported disease/trait text, not an ontology identifier."),
    Column("INITIAL SAMPLE SIZE", "initial_sample_size", "VARCHAR", "Free-text sample size/ancestry description."),
    Column("REPLICATION SAMPLE SIZE", "replication_sample_size", "VARCHAR", "Free-text replication description."),
    Column("REGION", "region", "VARCHAR", "Cytogenetic region(s), preserved as source text."),
    Column("CHR_ID", "chromosome", "VARCHAR", "Chromosome value(s), not necessarily a single integer."),
    Column("CHR_POS", "chromosome_position", "VARCHAR", "Position value(s); multi-variant records remain text."),
    Column("REPORTED GENE(S)", "reported_genes", "VARCHAR", "Author-reported genes; may contain multiple names."),
    Column(
        "MAPPED_GENE",
        "mapped_gene",
        "VARCHAR",
        "Catalog-mapped gene text; may list overlapping or flanking genes. Not a scalar gene identifier.",
    ),
    Column(
        "UPSTREAM_GENE_ID", "upstream_gene_id", "VARCHAR", "Source upstream gene identifier(s); namespace retained."
    ),
    Column(
        "DOWNSTREAM_GENE_ID",
        "downstream_gene_id",
        "VARCHAR",
        "Source downstream gene identifier(s); namespace retained.",
    ),
    Column("SNP_GENE_IDS", "snp_gene_ids", "VARCHAR", "Source overlapping gene identifier(s); may be multivalued."),
    Column("UPSTREAM_GENE_DISTANCE", "upstream_gene_distance", "BIGINT", "Source upstream distance in base pairs."),
    Column(
        "DOWNSTREAM_GENE_DISTANCE", "downstream_gene_distance", "BIGINT", "Source downstream distance in base pairs."
    ),
    Column(
        "STRONGEST SNP-RISK ALLELE",
        "strongest_snp_risk_allele",
        "VARCHAR",
        "Variant/risk-allele text; '?' can denote unknown allele. Haplotypes remain unsplit.",
    ),
    Column("SNPS", "snps", "VARCHAR", "Source variant text; a haplotype can contain multiple rsIDs."),
    Column("MERGED", "merged", "BIGINT", "Source merge flag (0/1); not converted to Boolean."),
    Column("SNP_ID_CURRENT", "snp_id_current", "VARCHAR", "Current SNP identifier as supplied; no rs prefix is added."),
    Column("CONTEXT", "context", "VARCHAR", "Source predicted functional consequence annotation."),
    Column("INTERGENIC", "intergenic", "BIGINT", "Source intergenic flag (0/1); not converted to Boolean."),
    Column(
        "RISK ALLELE FREQUENCY",
        "risk_allele_frequency",
        "VARCHAR",
        "Reported allele/haplotype frequency text; not cast to a number.",
    ),
    Column("P-VALUE", "p_value", "DOUBLE", "Reported Catalog p-value, with the source precision unchanged."),
    Column("PVALUE_MLOG", "pvalue_mlog", "DOUBLE", "Source negative log10 p-value; not recomputed."),
    Column("P-VALUE (TEXT)", "pvalue_text", "VARCHAR", "Context attached to the p-value, such as a subgroup."),
    Column(
        "OR or BETA",
        "or_or_beta",
        "DOUBLE",
        "Combined source effect value: odds ratio OR beta. Do not assume one effect type or split it.",
    ),
    Column("95% CI (TEXT)", "ci_95_text", "VARCHAR", "Source confidence-interval/unit text; not parsed."),
    Column(
        "PLATFORM [SNPS PASSING QC]",
        "platform_snps_passing_qc",
        "VARCHAR",
        "Platform and QC/imputation description; not a standalone numeric SNP count.",
    ),
    Column("CNV", "cnv", "VARCHAR", "Source copy-number-variation flag text."),
    Column(
        "MAPPED_TRAIT", "mapped_trait", "VARCHAR", "Mapped trait label(s); source multivalue formatting is retained."
    ),
    Column(
        "MAPPED_TRAIT_URI",
        "mapped_trait_uri",
        "VARCHAR",
        "Mapped ontology URI(s); no descendant expansion, label matching or namespace conversion is applied.",
    ),
    Column(
        "STUDY ACCESSION", "accession_id", "VARCHAR", "Catalog study accession; shared join key across the three views."
    ),
    Column("GENOTYPING TECHNOLOGY", "genotyping_technology", "VARCHAR", "Source technology description(s)."),
    Column("ASSOCIATION COUNT", "association_count", "BIGINT", "Source association count; NOT SNPs passing QC."),
    Column("COHORT", "cohort", "VARCHAR", "Source discovery-cohort text; multiple cohorts are not split."),
    Column(
        "FULL SUMMARY STATISTICS",
        "full_summary_statistics",
        "BOOLEAN",
        "Availability flag only; these views do not contain the genome-wide summary-statistics files.",
    ),
    Column(
        "SUMMARY STATS LOCATION",
        "summary_stats_location",
        "VARCHAR",
        "Source file location; not fetched by this catalog.",
    ),
    Column("GXE", "gxe", "BOOLEAN", "Source genotype-by-environment interaction study flag."),
    Column("INITIAL SAMPLE DESCRIPTION", "initial_sample_description", "VARCHAR", "Source initial-sample description."),
    Column(
        "REPLICATION SAMPLE DESCRIPTION", "replication_sample_description", "VARCHAR", "Source replication description."
    ),
    Column("STAGE", "stage", "VARCHAR", "Source sample stage, initial or replication in this release."),
    Column(
        "NUMBER OF INDIVDUALS",
        "number_of_individuals",
        "BIGINT",
        "Individuals in this sample row; source header typo retained in mapping.",
    ),
    Column(
        "BROAD ANCESTRAL CATEGORY",
        "broad_ancestral_category",
        "VARCHAR",
        "Source ancestry category text; not a newly inferred category.",
    ),
    Column("COUNTRY OF ORIGIN", "country_of_origin", "VARCHAR", "Source country-of-origin text."),
    Column("COUNTRY OF RECRUITMENT", "country_of_recruitment", "VARCHAR", "Source recruitment-country text."),
    Column(
        "ADDITONAL ANCESTRY DESCRIPTION",
        "additional_ancestry_description",
        "VARCHAR",
        "Additional source ancestry text; source header typo retained in mapping.",
    ),
]
_BY_SOURCE = {column.source: column for column in _FIELDS}


def _columns(*names: str) -> tuple[Column, ...]:
    return tuple(_BY_SOURCE[name] for name in names)


_PUBLICATION = (
    "DATE ADDED TO CATALOG",
    "PUBMEDID",
    "FIRST AUTHOR",
    "DATE",
    "JOURNAL",
    "LINK",
    "STUDY",
    "DISEASE/TRAIT",
    "INITIAL SAMPLE SIZE",
    "REPLICATION SAMPLE SIZE",
)

DATASETS = (
    Dataset(
        "associations",
        "7918ec608275b845b87c386478bdb1f4e549234c",
        "gwas-catalog-associations-r2026-03-17.parquet",
        "10f6d6fdd6237b32d6277d59ecd3220fe6b0aa81cd31334c65f29d480fbcf90b",
        68_633_383,
        _columns(
            *_PUBLICATION,
            "REGION",
            "CHR_ID",
            "CHR_POS",
            "REPORTED GENE(S)",
            "MAPPED_GENE",
            "UPSTREAM_GENE_ID",
            "DOWNSTREAM_GENE_ID",
            "SNP_GENE_IDS",
            "UPSTREAM_GENE_DISTANCE",
            "DOWNSTREAM_GENE_DISTANCE",
            "STRONGEST SNP-RISK ALLELE",
            "SNPS",
            "MERGED",
            "SNP_ID_CURRENT",
            "CONTEXT",
            "INTERGENIC",
            "RISK ALLELE FREQUENCY",
            "P-VALUE",
            "PVALUE_MLOG",
            "P-VALUE (TEXT)",
            "OR or BETA",
            "95% CI (TEXT)",
            "PLATFORM [SNPS PASSING QC]",
            "CNV",
            "MAPPED_TRAIT",
            "MAPPED_TRAIT_URI",
            "STUDY ACCESSION",
            "GENOTYPING TECHNOLOGY",
        ),
        "Curated association rows, not the full genome-wide summary statistics. "
        "Many rows may share a study or variant. No official association_id is present.",
    ),
    Dataset(
        "studies",
        "3542caab46a3a27667f198ed6f05e71c45a96618",
        "gwas-catalog-studies-r2026-03-17.parquet",
        "0d345722becb42b43d31058c8e8650c9f5788d2e88053e96187d4c3ef414333a",
        10_224_509,
        _columns(
            *_PUBLICATION,
            "PLATFORM [SNPS PASSING QC]",
            "ASSOCIATION COUNT",
            "MAPPED_TRAIT",
            "MAPPED_TRAIT_URI",
            "STUDY ACCESSION",
            "GENOTYPING TECHNOLOGY",
            "COHORT",
            "FULL SUMMARY STATISTICS",
            "SUMMARY STATS LOCATION",
            "GXE",
        ),
        "Study metadata. accession_id is unique and non-null in the verified pinned files. "
        "A publication can contain several studies; not every study has an association row.",
    ),
    Dataset(
        "samples",
        "f9cc5e6ec7cf2a870d29490847fdeb4b709c64dc",
        "gwas-catalog-ancestry-r2026-03-17.parquet",
        "917fa2b417a48c480daaee4546cce7d5113a349d590561814d45338bb4aa5054",
        3_387_704,
        _columns(
            "STUDY ACCESSION",
            "PUBMEDID",
            "FIRST AUTHOR",
            "DATE",
            "INITIAL SAMPLE DESCRIPTION",
            "REPLICATION SAMPLE DESCRIPTION",
            "STAGE",
            "NUMBER OF INDIVDUALS",
            "BROAD ANCESTRAL CATEGORY",
            "COUNTRY OF ORIGIN",
            "COUNTRY OF RECRUITMENT",
            "ADDITONAL ANCESTRY DESCRIPTION",
        ),
        "Sample/ancestry rows by study and stage. Multiple rows can share a study; "
        "rows are not globally unique people and summing them need not count unique participants.",
    ),
)

CATALOG_NOTES = (
    "These declarations describe the columns of runtime views, using empty-table DDL for reference.",
    "Data is attached by register_catalog; executing this file alone does not load any source data.",
    "All source values, row multiplicity and DuckDB types are preserved; only column names are aliased.",
    "SQL NULL and literal strings such as NA, NR and ? remain distinct; no global missing-value conversion.",
    "Join on accession_id. Joining associations to samples multiplies rows within each study; "
    "use EXISTS or aggregate at the intended level when appropriate.",
    "Do not join on paper title or assume pubmed_id identifies a single study.",
    "No manufactured association IDs, variant/gene token tables, ontology closure or extended gene set.",
    "mapped_gene and snps are raw potentially multivalued text; substring matching is not exact identifier matching.",
    "Column names are not a claim of REST v2 compatibility. Combined OR/beta and raw multivalues stay combined.",
    "Ontology and genome annotations are those in these source files; the current online API can differ.",
)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def configured_sources() -> dict[str, str]:
    """Resolve a local directory or S3 release prefix without involving the model.

    No network calls, credentials, release selection or fallback are performed.
    All locations still refer to the same pinned official filenames.
    """
    root = os.environ.get(DATA_ROOT_ENV)
    if root is None:
        root = str(DEFAULT_DATA_ROOT)
    if not root.strip():
        raise ValueError(f"{DATA_ROOT_ENV} cannot be empty")
    parsed = urlsplit(root)
    if parsed.scheme == "s3":
        if not parsed.netloc or parsed.username is not None or ":" in parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError(f"{DATA_ROOT_ENV} must be an s3://bucket/release-prefix URI")
        prefix = root.rstrip("/")
        return {dataset.name: f"{prefix}/{dataset.filename}" for dataset in DATASETS}
    if parsed.scheme:
        raise ValueError(f"{DATA_ROOT_ENV} supports only a local directory or an s3:// prefix")
    directory = Path(root).expanduser().resolve()
    return {dataset.name: str(directory / dataset.filename) for dataset in DATASETS}


def register_catalog(connection: duckdb.DuckDBPyConnection, sources: Mapping[str, str] | None = None) -> None:
    """Register all three views after validating every source column and type.

    Locations are harness configuration, not agent input; they may be local
    files or S3 URIs. S3 extensions/credentials must already be configured.
    This checks structure, not remote object hashes, SQL isolation or permissions.
    The caller must provide the verified immutable snapshot.
    When sources is omitted, GWAS_CATALOG_SQL_DATA_ROOT selects the location.
    """
    if sources is None:
        sources = configured_sources()
    expected_names = {dataset.name for dataset in DATASETS}
    if set(sources) != expected_names:
        raise ValueError(f"expected exactly these datasets: {sorted(expected_names)}")
    for dataset in DATASETS:
        if not isinstance(sources[dataset.name], str) or not sources[dataset.name]:
            raise ValueError(f"{dataset.name}: source location must be a nonempty string")
        expected = [(column.source, column.type) for column in dataset.columns]
        actual = [
            (row[0], row[1])
            for row in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [sources[dataset.name]]).fetchall()
        ]
        if actual != expected:
            raise ValueError(
                f"{dataset.name}: source columns/types/order differ from the pinned mapping; "
                f"expected {expected!r}, received {actual!r}"
            )

    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute("CREATE SCHEMA IF NOT EXISTS gwas")
        for dataset in DATASETS:
            relation = f"gwas.{quote_identifier(dataset.name)}"
            projection = ", ".join(
                f"{quote_identifier(column.source)} AS {quote_identifier(column.name)}" for column in dataset.columns
            )
            connection.execute(
                f"CREATE VIEW {relation} AS SELECT {projection} FROM read_parquet({_literal(sources[dataset.name])})"
            )
            connection.execute(f"COMMENT ON VIEW {relation} IS {_literal(dataset.description)}")
            for column in dataset.columns:
                connection.execute(
                    f"COMMENT ON COLUMN {relation}.{quote_identifier(column.name)} IS {_literal(column.comment)}"
                )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


def catalog_schema(connection: duckdb.DuckDBPyConnection) -> str:
    """Export actual catalog columns/comments as a compact agent schema file."""
    lines = [
        "-- GWAS Catalog SQL schema; generated from registered DuckDB views.",
        f"-- Source release: {RELEASE}; mapping version: {SCHEMA_VERSION}.",
        f"-- Field reference: {FILE_HEADERS_URL}",
        *(f"-- {note}" for note in CATALOG_NOTES),
        "",
        "CREATE SCHEMA IF NOT EXISTS gwas;",
    ]
    for dataset in DATASETS:
        rows = connection.execute(
            """SELECT column_name, data_type, comment FROM duckdb_columns()
               WHERE schema_name = 'gwas' AND table_name = ? ORDER BY column_index""",
            [dataset.name],
        ).fetchall()
        expected = [(column.name, column.type, column.comment) for column in dataset.columns]
        if rows != expected:
            raise ValueError(f"{dataset.name}: registered catalog differs from the field mapping")
        lines.extend(["", f"-- {dataset.description}", f"CREATE TABLE gwas.{dataset.name} ("])
        for index, (name, data_type, comment) in enumerate(rows):
            comma = "," if index + 1 < len(rows) else ""
            lines.append(f"    {quote_identifier(name)} {data_type}{comma} -- {comment}")
        lines.append(");")
    return "\n".join(lines) + "\n"
