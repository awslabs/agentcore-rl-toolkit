"""Download the environment data from its public upstreams and build the files the tool service reads.

    python prepare_data.py [--out data] [--only STEP ...]

Steps, in order: gwas, seqqa, screen, gene_sets, interactions, disease, clinvar. Downloads go to <out>/raw/ and are
reused when present; delete that directory once the build is done.
"""

import argparse
import ast
import collections
import csv
import gzip
import hashlib
import io
import json
import re
import sqlite3
import tarfile
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO

NCBI = "https://ftp.ncbi.nlm.nih.gov"
DATA_LAKE = "https://biomni-release.s3.amazonaws.com/data_lake"
MSIGDB = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ORCS = (
    "https://downloads.thebiogrid.org/Download/BioGRID-ORCS/Release-Archive/BIOGRID-ORCS-2.0.18/"
    "BIOGRID-ORCS-ALL-homo_sapiens-2.0.18.screens.tar.gz"
)
ORCS_INDEX = "BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt"
MONDO = "https://github.com/monarch-initiative/mondo/releases/download/v2026-08-04/mondo.obo"
HPO = "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2026-06-23/genes_to_disease.txt"
ENSEMBL_PEP = "https://ftp.ensembl.org/pub/release-116/fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz"
MANE = f"{NCBI}/refseq/MANE/MANE_human/release_1.5/MANE.GRCh38.v1.5.summary.txt.gz"
REFSEQ_PROTEINS = f"{NCBI}/refseq/H_sapiens/mRNA_Prot/"  # current release: no archive
CLINVAR_SUMMARY = f"{NCBI}/pub/clinvar/tab_delimited/archive/variant_summary_2026-09.txt.gz"
CLINVAR_HGVS = f"{NCBI}/pub/clinvar/tab_delimited/hgvs4variation.txt.gz"  # current release: no archive
SEQQA_REFS = {  # file name -> (accession, efetch rettype)
    "EcoliK12-NC_000913.3.gbwithparts": ("NC_000913.3", "gbwithparts"),
    "pUC19-M77789.2.gb": ("M77789.2", "gb"),
}
# Gene-set tables: name -> Biomni data lake parquet, or an MSigDB GMT with its collection.
DATA_LAKE_GENE_SETS = {
    "mousemine_m5": "mousemine_m5_ontology_geneset.parquet",
    "msigdb_c6": "msigdb_human_c6_oncogenic_signature_geneset.parquet",
    "msigdb_c7": "msigdb_human_c7_immunologic_signature_geneset.parquet",
}
GMT_GENE_SETS = {
    "msigdb_gtrd": ("c3.tft.gtrd.v2023.2.Hs.symbols.gmt", "C3:TFT:GTRD"),
    "msigdb_mirdb": ("c3.mir.mirdb.v2023.2.Hs.symbols.gmt", "C3:MIR:MIRDB"),
    "ensembl_cytoband": ("c1.all.v2023.2.Hs.symbols.gmt", "C1"),
}
HGVS_ACCESSION = re.compile(r"^([A-Z]{2}_[0-9]+)(?:\.[0-9]+)?:")
DOWNLOAD_ATTEMPTS = 5
HGVS_CHANGE = re.compile(r"^p\.\(?([A-Z][a-z]{2}[0-9]+[A-Z][a-z]{2})\)?$")


def download(url: str, raw: Path, name: str | None = None) -> Path:
    path = raw / (name or url.rsplit("/", 1)[-1])
    if path.exists():
        return path
    print(f"downloading {url}", flush=True)
    partial = path.with_name(path.name + ".partial")
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, stream=True, timeout=(30, 300), headers=headers) as response:
                response.raise_for_status()
                resumed = offset and response.status_code == 206
                with partial.open("ab" if resumed else "wb") as stream:
                    for chunk in response.iter_content(1 << 20):
                        stream.write(chunk)
            break
        except requests.RequestException as exc:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            print(f"  attempt {attempt} failed ({type(exc).__name__}); resuming", flush=True)
    partial.rename(path)
    return path


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=OFF")
    return con


# --------------------------------------------------------------------------- gwas, seqqa


def build_gwas(out: Path, raw: Path) -> None:
    """The three official GWAS Catalog files; the SQL backend checks their sizes and sha256 at startup."""
    from harness.gwas.clients.gwas_catalog_sql.catalog import DATASETS, RELEASE

    target = out / "gwas_catalog" / RELEASE
    target.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        download(dataset.source_url, target, dataset.filename)


def build_seqqa(out: Path, raw: Path) -> None:
    target = out / "seqqa" / "refs"
    target.mkdir(parents=True, exist_ok=True)
    for name, (accession, rettype) in SEQQA_REFS.items():
        download(f"{EFETCH}?db=nuccore&id={accession}&rettype={rettype}&retmode=text", target, name)


# --------------------------------------------------------------------------- screen


def build_screen(out: Path, raw: Path) -> None:
    """BioGRID ORCS 2.0.18 human screens: the screen index and every screen's scores, without the HIT column."""

    def column(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    con = connect(out / "screen" / "orcs_nohit.sqlite")
    with tarfile.open(download(ORCS, raw), "r:gz") as archive:
        index = archive.extractfile(archive.getmember(ORCS_INDEX))
        rows = list(csv.DictReader(io.TextIOWrapper(index, errors="ignore"), delimiter="\t"))
        fields = list(rows[0])
        con.execute("CREATE TABLE screens (%s)" % ", ".join(f"{column(f)} TEXT" for f in fields))
        con.executemany(
            "INSERT INTO screens VALUES (%s)" % ", ".join("?" * len(fields)),
            [tuple((row.get(f) or "").strip() for f in fields) for row in rows],
        )
        con.execute(
            "CREATE TABLE scores (screen_id TEXT, symbol TEXT, aliases TEXT, "
            "score_1 TEXT, score_2 TEXT, score_3 TEXT, score_4 TEXT, score_5 TEXT)"
        )
        for member in archive:
            if not member.name.endswith(".screen.tab.txt"):
                continue
            reader = csv.DictReader(io.TextIOWrapper(archive.extractfile(member), errors="ignore"), delimiter="\t")
            con.executemany(
                "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        (row.get("#SCREEN_ID") or "").strip(),
                        (row.get("OFFICIAL_SYMBOL") or "").strip().upper(),
                        (row.get("ALIASES") or "").strip(),
                        *(row.get(f"SCORE.{i}") for i in range(1, 6)),
                    )
                    for row in reader
                ),
            )
    con.execute("CREATE INDEX ix_screens_id ON screens(screen_id)")
    con.execute("CREATE INDEX ix_scores_screen_symbol ON scores(screen_id, symbol)")
    con.execute("CREATE INDEX ix_scores_symbol ON scores(symbol)")
    con.commit()
    con.close()


# --------------------------------------------------------------------------- dbqa: gene sets, interactions, disease


def members(value) -> list[str]:
    """A source's member list, stored as a JSON or Python list literal."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = ast.literal_eval(value)
    if not isinstance(value, list):
        raise ValueError(f"expected a list of members, got {type(value).__name__}")
    return sorted({str(item) for item in value})


def source_table(con: sqlite3.Connection, name: str, frame: pd.DataFrame, key: str, member_field: str) -> None:
    """The source's own fields as a table, plus a <name>_members table of (record_id, gene_symbol)."""
    records, associations = [], []
    for values in frame.itertuples(index=False, name=None):
        row = dict(zip(frame.columns, values, strict=True))
        records.append(
            tuple(json.dumps(v) if isinstance(v, list | dict) else None if pd.isna(v) else str(v) for v in values)
        )
        associations.extend((str(row[key]), gene) for gene in members(row[member_field]))
    columns = ", ".join(f'"{c}"' for c in frame.columns)
    con.execute(f'CREATE TABLE "{name}" ({", ".join(f'"{c}" TEXT' for c in frame.columns)})')
    con.executemany(f'INSERT INTO "{name}" ({columns}) VALUES ({", ".join("?" * len(frame.columns))})', records)
    con.execute(f'CREATE INDEX "{name}_key" ON "{name}" ("{key}" COLLATE NOCASE)')
    table = f"{name}_members"
    con.execute(f'CREATE TABLE "{table}" (record_id TEXT NOT NULL, gene_symbol TEXT NOT NULL)')
    con.executemany(f'INSERT INTO "{table}" VALUES (?, ?)', associations)
    con.execute(f'CREATE INDEX "{table}_record" ON "{table}" (record_id COLLATE NOCASE, gene_symbol COLLATE NOCASE)')
    con.execute(f'CREATE INDEX "{table}_exact" ON "{table}" (record_id, gene_symbol)')
    con.execute(f'CREATE INDEX "{table}_gene" ON "{table}" (gene_symbol COLLATE NOCASE)')


def gmt_frame(path: Path, collection: str) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        name, url, *genes = line.split("\t")
        rows.append(
            {"chromosome_id": name, "collection": collection, "geneSymbols": json.dumps(genes), "msigdbURL": url}
        )
    return pd.DataFrame(rows)


def build_gene_sets(out: Path, raw: Path) -> None:
    con = connect(out / "dbqa" / "gene_sets.sqlite")
    for name, filename in DATA_LAKE_GENE_SETS.items():
        frame = pd.read_parquet(download(f"{DATA_LAKE}/{filename}", raw))
        source_table(con, name, frame, "chromosome_id", "geneSymbols")
    for name, (filename, collection) in GMT_GENE_SETS.items():
        frame = gmt_frame(download(f"{MSIGDB}/{filename}", raw), collection)
        source_table(con, name, frame, "chromosome_id", "geneSymbols")
    con.commit()
    con.close()


def build_interactions(out: Path, raw: Path) -> None:
    con = connect(out / "dbqa" / "interactions.sqlite")
    frame = pd.read_parquet(download(f"{DATA_LAKE}/Virus-Host_PPI_P-HIPSTER_2020.parquet", raw))
    source_table(con, "phipster_2020", frame, "Viral Protien", "Genes")
    con.commit()
    con.close()


def mondo_terms(path: Path):
    """(id, name, synonyms, xrefs) of every MONDO class in the OBO release, obsolete ones included."""
    term = None
    for line in path.read_text().splitlines() + ["[Term]"]:
        if line.startswith("["):
            if term and term["id"].startswith("MONDO:"):
                yield term["id"], term["name"], term["synonyms"], term["xrefs"]
            term = None
            if line == "[Term]":
                term = {"id": "", "name": None, "synonyms": [], "xrefs": []}
        elif term is not None and ": " in line:
            tag, value = line.split(": ", 1)
            if tag == "id":
                term["id"] = value
            elif tag == "name":
                term["name"] = value
            elif tag == "synonym":
                term["synonyms"].append(value.split('"')[1])
            elif tag == "xref":
                term["xrefs"].append(value.split(" ", 1)[0])


def build_disease(out: Path, raw: Path) -> None:
    con = connect(out / "dbqa" / "disease.sqlite")
    source_table(con, "disgenet", pd.read_parquet(download(f"{DATA_LAKE}/DisGeNET.parquet", raw)), "Disorder", "Genes")
    con.execute("CREATE TABLE mondo (mondo_id TEXT PRIMARY KEY, name TEXT)")
    con.execute("CREATE TABLE mondo_aliases (mondo_id TEXT, alias TEXT)")
    con.execute("CREATE TABLE mondo_xrefs (mondo_id TEXT, xref TEXT)")
    for mondo_id, name, synonyms, xrefs in mondo_terms(download(MONDO, raw)):
        con.execute("INSERT INTO mondo VALUES (?, ?)", (mondo_id, name))
        con.executemany("INSERT INTO mondo_aliases VALUES (?, ?)", [(mondo_id, a) for a in [name, *synonyms]])
        con.executemany("INSERT INTO mondo_xrefs VALUES (?, ?)", [(mondo_id, x) for x in xrefs])
    con.execute("CREATE INDEX mondo_alias ON mondo_aliases (alias COLLATE NOCASE)")
    con.execute("CREATE INDEX mondo_xref ON mondo_xrefs (xref)")
    con.execute(
        "CREATE TABLE hpo_omim_genes (disease_id TEXT, gene_symbol TEXT, association_type TEXT, ncbi_gene_id TEXT)"
    )
    with download(HPO, raw).open() as stream:
        next(stream)
        con.executemany(
            "INSERT INTO hpo_omim_genes VALUES (?, ?, ?, ?)",
            (
                (fields[3], fields[1], fields[2], fields[0])
                for line in stream
                if len(fields := line.rstrip("\n").split("\t")) >= 5 and fields[3].startswith("OMIM:")
            ),
        )
    con.execute("CREATE INDEX hpo_disease ON hpo_omim_genes (disease_id, gene_symbol COLLATE NOCASE)")
    con.commit()
    con.close()


# --------------------------------------------------------------------------- dbqa: ClinVar and protein accessions


def protein_accessions(raw: Path) -> dict:
    """sha256 of each human protein sequence -> its Ensembl 116 and RefSeq protein accessions."""
    by_hash = collections.defaultdict(lambda: {"ensembl_proteins": set(), "refseq_proteins": set()})
    ensp_hash = {}
    with gzip.open(download(ENSEMBL_PEP, raw), "rt") as stream:
        for record in SeqIO.parse(stream, "fasta"):
            digest = hashlib.sha256(str(record.seq).encode()).hexdigest()
            stable = record.id.split(".")[0]
            by_hash[digest]["ensembl_proteins"].add(stable)
            ensp_hash[stable] = digest
    mane = pd.read_csv(download(MANE, raw), sep="\t", dtype=str)
    for row in mane.itertuples(index=False):
        if isinstance(row.Ensembl_prot, str) and isinstance(row.RefSeq_prot, str):
            digest = ensp_hash.get(row.Ensembl_prot.split(".")[0])
            if digest:
                by_hash[digest]["refseq_proteins"].add(row.RefSeq_prot.split(".")[0])
    listing = requests.get(REFSEQ_PROTEINS, timeout=60).text
    for name in sorted(set(re.findall(r"human\.\d+\.protein\.faa\.gz", listing))):
        with gzip.open(download(REFSEQ_PROTEINS + name, raw), "rt") as stream:
            for record in SeqIO.parse(stream, "fasta"):
                accession = record.id.split(".")[0]
                if accession.startswith("NP_"):
                    by_hash[hashlib.sha256(str(record.seq).encode()).hexdigest()]["refseq_proteins"].add(accession)
    return {digest: {k: sorted(v) for k, v in item.items()} for digest, item in by_hash.items()}


def build_clinvar(out: Path, raw: Path) -> None:
    """ClinVar protein changes on the RefSeq proteins above, with each variant's gene, classification and review."""
    mapping = protein_accessions(raw)
    (out / "dbqa").mkdir(parents=True, exist_ok=True)
    (out / "dbqa" / "protein_accessions.json").write_text(json.dumps(mapping, sort_keys=True, separators=(",", ":")))
    wanted = {accession for item in mapping.values() for accession in item["refseq_proteins"]}
    matches = set()
    with gzip.open(download(CLINVAR_HGVS, raw), "rt") as stream:
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            if line.startswith("#") or len(parts) <= 9:
                continue
            accession, change = HGVS_ACCESSION.match(parts[8]), HGVS_CHANGE.match(parts[9])
            if accession and change and accession.group(1) in wanted:
                matches.add((accession.group(1), change.group(1), parts[2]))
    needed = {variation_id for _, _, variation_id in matches}
    clinical = {}
    with gzip.open(download(CLINVAR_SUMMARY, raw), "rt") as stream:
        header = stream.readline().lstrip("#").rstrip("\n").split("\t")
        ix = [header.index(k) for k in ("VariationID", "GeneSymbol", "ClinicalSignificance", "ReviewStatus")]
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            if parts[ix[0]] in needed:
                clinical.setdefault(parts[ix[0]], set()).add(tuple(parts[i] for i in ix[1:]))
    con = connect(out / "dbqa" / "clinvar.sqlite")
    con.execute(
        "CREATE TABLE variant (accession TEXT, change TEXT, variation_id TEXT, "
        "gene TEXT, classification TEXT, review_status TEXT)"
    )
    con.executemany(
        "INSERT INTO variant VALUES (?, ?, ?, ?, ?, ?)",
        (
            (accession, change, variation_id, *details)
            for accession, change, variation_id in sorted(matches)
            for details in sorted(clinical.get(variation_id, {("", "", "")}))
        ),
    )
    con.execute("CREATE INDEX variant_query ON variant(accession, change)")
    con.commit()
    con.close()


STEPS = {
    "gwas": build_gwas,
    "seqqa": build_seqqa,
    "screen": build_screen,
    "gene_sets": build_gene_sets,
    "interactions": build_interactions,
    "disease": build_disease,
    "clinvar": build_clinvar,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--only", nargs="+", choices=STEPS)
    args = parser.parse_args()
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in args.only or STEPS:
        print(f"== {name}", flush=True)
        STEPS[name](args.out, raw)


if __name__ == "__main__":
    main()
