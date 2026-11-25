-- GWAS Catalog SQL schema; generated from registered DuckDB views.
-- Source release: 2026-03-17; mapping version: 1.
-- Field reference: https://www.ebi.ac.uk/gwas/docs/fileheaders
-- These declarations describe the columns of runtime views, using empty-table DDL for reference.
-- Data is attached by register_catalog; executing this file alone does not load any source data.
-- All source values, row multiplicity and DuckDB types are preserved; only column names are aliased.
-- SQL NULL and literal strings such as NA, NR and ? remain distinct; no global missing-value conversion.
-- Join on accession_id. Joining associations to samples multiplies rows within each study; use EXISTS or aggregate at the intended level when appropriate.
-- Do not join on paper title or assume pubmed_id identifies a single study.
-- No manufactured association IDs, variant/gene token tables, ontology closure or extended gene set.
-- mapped_gene and snps are raw potentially multivalued text; substring matching is not exact identifier matching.
-- Column names are not a claim of REST v2 compatibility. Combined OR/beta and raw multivalues stay combined.
-- Ontology and genome annotations are those in these source files; the current online API can differ.

CREATE SCHEMA IF NOT EXISTS gwas;

-- Curated association rows, not the full genome-wide summary statistics. Many rows may share a study or variant. No official association_id is present.
CREATE TABLE gwas.associations (
    "date_added_to_catalog" DATE, -- Source: DATE ADDED TO CATALOG. Date published in the Catalog.
    "pubmed_id" BIGINT, -- Source: PUBMEDID. PubMed publication identifier; several studies can share it.
    "first_author" VARCHAR, -- Source: FIRST AUTHOR. First author as supplied.
    "publication_date" DATE, -- Source: DATE. Publication date, not the Catalog release date.
    "journal" VARCHAR, -- Source: JOURNAL. Journal name as supplied.
    "publication_link" VARCHAR, -- Source: LINK. Publication link as supplied; a URL scheme may be absent.
    "study_title" VARCHAR, -- Source: STUDY. Paper title; accession_id identifies the Catalog study.
    "disease_trait" VARCHAR, -- Source: DISEASE/TRAIT. Reported disease/trait text, not an ontology identifier.
    "initial_sample_size" VARCHAR, -- Source: INITIAL SAMPLE SIZE. Free-text sample size/ancestry description.
    "replication_sample_size" VARCHAR, -- Source: REPLICATION SAMPLE SIZE. Free-text replication description.
    "region" VARCHAR, -- Source: REGION. Cytogenetic region(s), preserved as source text.
    "chromosome" VARCHAR, -- Source: CHR_ID. Chromosome value(s), not necessarily a single integer.
    "chromosome_position" VARCHAR, -- Source: CHR_POS. Position value(s); multi-variant records remain text.
    "reported_genes" VARCHAR, -- Source: REPORTED GENE(S). Author-reported genes; may contain multiple names.
    "mapped_gene" VARCHAR, -- Source: MAPPED_GENE. Catalog-mapped gene text; may list overlapping or flanking genes. Not a scalar gene identifier.
    "upstream_gene_id" VARCHAR, -- Source: UPSTREAM_GENE_ID. Source upstream gene identifier(s); namespace retained.
    "downstream_gene_id" VARCHAR, -- Source: DOWNSTREAM_GENE_ID. Source downstream gene identifier(s); namespace retained.
    "snp_gene_ids" VARCHAR, -- Source: SNP_GENE_IDS. Source overlapping gene identifier(s); may be multivalued.
    "upstream_gene_distance" BIGINT, -- Source: UPSTREAM_GENE_DISTANCE. Source upstream distance in base pairs.
    "downstream_gene_distance" BIGINT, -- Source: DOWNSTREAM_GENE_DISTANCE. Source downstream distance in base pairs.
    "strongest_snp_risk_allele" VARCHAR, -- Source: STRONGEST SNP-RISK ALLELE. Variant/risk-allele text; '?' can denote unknown allele. Haplotypes remain unsplit.
    "snps" VARCHAR, -- Source: SNPS. Source variant text; a haplotype can contain multiple rsIDs.
    "merged" BIGINT, -- Source: MERGED. Source merge flag (0/1); not converted to Boolean.
    "snp_id_current" VARCHAR, -- Source: SNP_ID_CURRENT. Current SNP identifier as supplied; no rs prefix is added.
    "context" VARCHAR, -- Source: CONTEXT. Source predicted functional consequence annotation.
    "intergenic" BIGINT, -- Source: INTERGENIC. Source intergenic flag (0/1); not converted to Boolean.
    "risk_allele_frequency" VARCHAR, -- Source: RISK ALLELE FREQUENCY. Reported allele/haplotype frequency text; not cast to a number.
    "p_value" DOUBLE, -- Source: P-VALUE. Reported Catalog p-value, with the source precision unchanged.
    "pvalue_mlog" DOUBLE, -- Source: PVALUE_MLOG. Source negative log10 p-value; not recomputed.
    "pvalue_text" VARCHAR, -- Source: P-VALUE (TEXT). Context attached to the p-value, such as a subgroup.
    "or_or_beta" DOUBLE, -- Source: OR or BETA. Combined source effect value: odds ratio OR beta. Do not assume one effect type or split it.
    "ci_95_text" VARCHAR, -- Source: 95% CI (TEXT). Source confidence-interval/unit text; not parsed.
    "platform_snps_passing_qc" VARCHAR, -- Source: PLATFORM [SNPS PASSING QC]. Platform and QC/imputation description; not a standalone numeric SNP count.
    "cnv" VARCHAR, -- Source: CNV. Source copy-number-variation flag text.
    "mapped_trait" VARCHAR, -- Source: MAPPED_TRAIT. Mapped trait label(s); source multivalue formatting is retained.
    "mapped_trait_uri" VARCHAR, -- Source: MAPPED_TRAIT_URI. Mapped ontology URI(s); no descendant expansion, label matching or namespace conversion is applied.
    "accession_id" VARCHAR, -- Source: STUDY ACCESSION. Catalog study accession; shared join key across the three views.
    "genotyping_technology" VARCHAR -- Source: GENOTYPING TECHNOLOGY. Source technology description(s).
);

-- Study metadata. accession_id is unique and non-null in the verified pinned files. A publication can contain several studies; not every study has an association row.
CREATE TABLE gwas.studies (
    "date_added_to_catalog" DATE, -- Source: DATE ADDED TO CATALOG. Date published in the Catalog.
    "pubmed_id" BIGINT, -- Source: PUBMEDID. PubMed publication identifier; several studies can share it.
    "first_author" VARCHAR, -- Source: FIRST AUTHOR. First author as supplied.
    "publication_date" DATE, -- Source: DATE. Publication date, not the Catalog release date.
    "journal" VARCHAR, -- Source: JOURNAL. Journal name as supplied.
    "publication_link" VARCHAR, -- Source: LINK. Publication link as supplied; a URL scheme may be absent.
    "study_title" VARCHAR, -- Source: STUDY. Paper title; accession_id identifies the Catalog study.
    "disease_trait" VARCHAR, -- Source: DISEASE/TRAIT. Reported disease/trait text, not an ontology identifier.
    "initial_sample_size" VARCHAR, -- Source: INITIAL SAMPLE SIZE. Free-text sample size/ancestry description.
    "replication_sample_size" VARCHAR, -- Source: REPLICATION SAMPLE SIZE. Free-text replication description.
    "platform_snps_passing_qc" VARCHAR, -- Source: PLATFORM [SNPS PASSING QC]. Platform and QC/imputation description; not a standalone numeric SNP count.
    "association_count" BIGINT, -- Source: ASSOCIATION COUNT. Source association count; NOT SNPs passing QC.
    "mapped_trait" VARCHAR, -- Source: MAPPED_TRAIT. Mapped trait label(s); source multivalue formatting is retained.
    "mapped_trait_uri" VARCHAR, -- Source: MAPPED_TRAIT_URI. Mapped ontology URI(s); no descendant expansion, label matching or namespace conversion is applied.
    "accession_id" VARCHAR, -- Source: STUDY ACCESSION. Catalog study accession; shared join key across the three views.
    "genotyping_technology" VARCHAR, -- Source: GENOTYPING TECHNOLOGY. Source technology description(s).
    "cohort" VARCHAR, -- Source: COHORT. Source discovery-cohort text; multiple cohorts are not split.
    "full_summary_statistics" BOOLEAN, -- Source: FULL SUMMARY STATISTICS. Availability flag only; these views do not contain the genome-wide summary-statistics files.
    "summary_stats_location" VARCHAR, -- Source: SUMMARY STATS LOCATION. Source file location; not fetched by this catalog.
    "gxe" BOOLEAN -- Source: GXE. Source genotype-by-environment interaction study flag.
);

-- Sample/ancestry rows by study and stage. Multiple rows can share a study; rows are not globally unique people and summing them need not count unique participants.
CREATE TABLE gwas.samples (
    "accession_id" VARCHAR, -- Source: STUDY ACCESSION. Catalog study accession; shared join key across the three views.
    "pubmed_id" BIGINT, -- Source: PUBMEDID. PubMed publication identifier; several studies can share it.
    "first_author" VARCHAR, -- Source: FIRST AUTHOR. First author as supplied.
    "publication_date" DATE, -- Source: DATE. Publication date, not the Catalog release date.
    "initial_sample_description" VARCHAR, -- Source: INITIAL SAMPLE DESCRIPTION. Source initial-sample description.
    "replication_sample_description" VARCHAR, -- Source: REPLICATION SAMPLE DESCRIPTION. Source replication description.
    "stage" VARCHAR, -- Source: STAGE. Source sample stage, initial or replication in this release.
    "number_of_individuals" BIGINT, -- Source: NUMBER OF INDIVDUALS. Individuals in this sample row; source header typo retained in mapping.
    "broad_ancestral_category" VARCHAR, -- Source: BROAD ANCESTRAL CATEGORY. Source ancestry category text; not a newly inferred category.
    "country_of_origin" VARCHAR, -- Source: COUNTRY OF ORIGIN. Source country-of-origin text.
    "country_of_recruitment" VARCHAR, -- Source: COUNTRY OF RECRUITMENT. Source recruitment-country text.
    "additional_ancestry_description" VARCHAR -- Source: ADDITONAL ANCESTRY DESCRIPTION. Additional source ancestry text; source header typo retained in mapping.
);
