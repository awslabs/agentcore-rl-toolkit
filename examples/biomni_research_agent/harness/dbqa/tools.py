"""Direct local evidence queries for DBQA. No answer or auxiliary-model path."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from functools import lru_cache

from ..common import online
from ..common.paths import DATA

INDEX = DATA / "dbqa"
PINNED = DATA / "dbqa"
SOURCES = {
    "query_gene_sets": (
        INDEX / "gene_sets.sqlite",
        {
            "mousemine_m5",
            "mousemine_m5_members",
            "msigdb_c6",
            "msigdb_c6_members",
            "msigdb_c7",
            "msigdb_c7_members",
            "msigdb_gtrd",
            "msigdb_gtrd_members",
            "msigdb_mirdb",
            "msigdb_mirdb_members",
            "ensembl_cytoband",
            "ensembl_cytoband_members",
        },
        "MouseMine M5 / MSigDB 2023.2 / Ensembl cytoband",
    ),
    "query_interactions": (
        INDEX / "interactions.sqlite",
        {"phipster_2020", "phipster_2020_members"},
        "P-HIPSter 2020",
    ),
    "query_disease_evidence": (
        INDEX / "disease.sqlite",
        {
            "disgenet",
            "disgenet_members",
            "mondo",
            "mondo_aliases",
            "mondo_xrefs",
            "hpo_omim_genes",
        },
        "DisGeNET / MONDO / HPO OMIM-linked genes",
    ),
    "query_clinvar_records": (PINNED / "clinvar.sqlite", {"variant"}, "ClinVar 2026-09-29 local snapshot"),
}
MAX_QUERY_CHARS = 8000
MAX_ROWS = 100
MAX_RESULT_BYTES = 48_000
MAX_QUERY_SECONDS = 15.0
MAX_SQL_STEPS = 2_000_000

AA3 = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
}


def _read_only_authorizer(tables: set[str]):
    def authorize(action: int, arg1: str | None, arg2: str | None, *_):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and arg1 in tables:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            # SQLite extension loading is disabled on the connection.
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    return authorize


def _sql_query(tool: str, query: str, offset: int = 0) -> dict:
    path, tables, source = SOURCES[tool]
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= MAX_QUERY_CHARS:
        return {"status": "invalid_query", "source": source}
    statement = query.strip().removesuffix(";").strip()
    if not re.match(r"(?is)^(SELECT|WITH)\b", statement):
        return {
            "status": "invalid_query",
            "source": source,
            "error": "Supply one read-only SELECT query.",
        }
    if type(offset) is not int or not 0 <= offset <= 100_000:  # noqa: E721
        return {"status": "invalid_offset", "source": source}
    started = time.monotonic()
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, timeout=1.0)
    except sqlite3.Error as exc:
        return {"status": "backend_error", "source": source, "error": str(exc)}
    steps = 0
    stopped = False

    def check_budget() -> int:
        nonlocal steps, stopped
        steps += 1000
        stopped = steps > MAX_SQL_STEPS or time.monotonic() - started > MAX_QUERY_SECONDS
        return int(stopped)

    try:
        con.execute("PRAGMA query_only=ON")
        con.set_authorizer(_read_only_authorizer(tables))
        con.set_progress_handler(check_budget, 1000)
        # Wrapping gives every query a common, explicit pagination contract.
        cursor = con.execute(
            f"SELECT * FROM ({statement}) AS submitted_query LIMIT ? OFFSET ?",
            (MAX_ROWS + 1, offset),
        )
        columns = [column[0] for column in cursor.description]
        fetched = cursor.fetchall()
        rows = []
        for values in fetched[:MAX_ROWS]:
            row = dict(zip(columns, values, strict=False))
            prospective = {
                "columns": columns,
                "rows": [*rows, row],
                "source": source,
                "status": "ok",
            }
            if len(json.dumps(prospective, ensure_ascii=False).encode()) > MAX_RESULT_BYTES:
                if not rows:
                    return {
                        "status": "result_too_large",
                        "source": source,
                        "error": "One row exceeds the result budget; select fewer columns.",
                    }
                break
            rows.append(row)
        return {
            "status": "ok",
            "source": source,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "offset": offset,
            "has_more": len(fetched) > len(rows),
            "next_offset": offset + len(rows) if len(fetched) > len(rows) else None,
            "seconds": round(time.monotonic() - started, 3),
        }
    except sqlite3.Error as exc:
        message = str(exc)
        if stopped:
            status = "query_budget_exceeded"
        elif isinstance(exc, sqlite3.OperationalError) and (
            "no such" in message
            or "syntax" in message
            or "misuse" in message
            or "ambiguous" in message
            or "only execute one" in message
            or "prohibited" in message
            or "not authorized" in message
            or "near " in message
            or "incomplete input" in message
            or "wrong number" in message
            or "unrecognized token" in message
        ):
            status = "query_error"
        else:
            status = "backend_error"
        return {"status": status, "source": source, "error": message}
    finally:
        con.close()


def query_gene_sets(query: str, offset: int = 0) -> dict:
    return _sql_query("query_gene_sets", query, offset)


def query_interactions(query: str, offset: int = 0) -> dict:
    return _sql_query("query_interactions", query, offset)


def query_disease_evidence(query: str, offset: int = 0) -> dict:
    result = _sql_query("query_disease_evidence", query, offset)
    result["omim_absence_is_negative_evidence"] = False
    return result


def query_clinvar_records(query: str, offset: int = 0) -> dict:
    return _sql_query("query_clinvar_records", query, offset)


@lru_cache(maxsize=1)
def _proteins() -> dict:
    return json.loads((PINNED / "protein_accessions.json").read_text())


def query_protein_records(sequence: str) -> dict:
    """Exact sequence is a model-written query to the fixed accession index."""
    if not isinstance(sequence, str) or not re.fullmatch(r"[A-Z]{20,20000}", sequence):
        return {"status": "invalid_sequence"}
    protein = _proteins().get(hashlib.sha256(sequence.encode()).hexdigest())
    return {
        "status": "ok" if protein else "sequence_not_found",
        "source": "Ensembl 116 / MANE 1.5 / RefSeq",
        "protein": protein or {},
    }


def compare_sequences(reference: str, sequences: list[str], start_position: int = 1) -> dict:
    if (
        not isinstance(reference, str)
        or not re.fullmatch(r"[A-Z]{20,20000}", reference)
        or not isinstance(sequences, list)
        or not 1 <= len(sequences) <= 8
        or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z]{20,20000}", s) for s in sequences)
        or type(start_position) is not int  # noqa: E721
        or not 1 <= start_position <= 20_000
    ):
        return {"status": "invalid_sequences"}
    results = []
    for sequence in sequences:
        same_length = len(reference) == len(sequence)
        differences = (
            [(i, old, new) for i, (old, new) in enumerate(zip(reference, sequence, strict=False), 1) if old != new]
            if same_length
            else []
        )
        visible = [
            {
                "position": i,
                "reference": old,
                "alternate": new,
                "one_letter": f"{old}{i}{new}",
                "three_letter": (f"{AA3[old]}{i}{AA3[new]}" if old in AA3 and new in AA3 else None),
            }
            for i, old, new in differences
            if i >= start_position
        ][:100]
        next_position = (
            visible[-1]["position"] + 1
            if (visible and any(i > visible[-1]["position"] for i, _, _ in differences))
            else None
        )
        results.append(
            {
                "same_length": same_length,
                "difference_count": len(differences) if same_length else None,
                "differences": visible,
                "next_position": next_position,
            }
        )
    return {"status": "ok", "results": results}


def _sql_spec(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description
            + (
                " Write one SQLite SELECT (WITH is allowed). "
                "Results include source, status and pagination; use offset to continue."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQLite SELECT query."},
                    "offset": {"type": "integer", "minimum": 0},
                },
                "required": ["query"],
            },
        },
    }


TOOL_SPECS = [
    _sql_spec(
        "query_gene_sets",
        "Fixed source tables mousemine_m5, msigdb_c6, msigdb_c7, "
        "msigdb_gtrd, msigdb_mirdb, ensembl_cytoband have native "
        "chromosome_id and geneSymbols columns. Each has a _members table "
        "(record_id, gene_symbol) for exact member queries. "
        "MSigDB tables use 2023.2; MouseMine is M5.",
    ),
    _sql_spec(
        "query_interactions",
        "P-HIPSter 2020 table phipster_2020 has native columns "
        '"Viral Protien", Genes; phipster_2020_members has '
        "(record_id, gene_symbol).",
    ),
    _sql_spec(
        "query_disease_evidence",
        "Tables: disgenet (Disorder, Genes), disgenet_members "
        "(record_id, gene_symbol), mondo (mondo_id, name), "
        "mondo_aliases (mondo_id, alias), mondo_xrefs (mondo_id, xref), "
        "hpo_omim_genes (disease_id, gene_symbol, association_type, ncbi_gene_id). "
        "HPO OMIM-linked associations are incomplete; a missing row does not "
        "establish absence of an OMIM disease association.",
    ),
    _sql_spec(
        "query_clinvar_records",
        "Fixed local ClinVar 2026-09-29 variant table has accession, change "
        "(three-letter protein HGVS, e.g. Ala123Val), variation_id, "
        "gene, classification, review_status. Match protein accession. "
        "This is a local protein-variant projection, not the whole online API.",
    ),
    {
        "type": "function",
        "function": {
            "name": "query_protein_records",
            "description": (
                "Find Ensembl 116, MANE 1.5 and RefSeq protein accessions "
                "for an exact amino-acid sequence in the fixed local index."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sequence": {"type": "string"}},
                "required": ["sequence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_sequences",
            "description": (
                "Compute amino-acid changes relative to a supplied reference "
                "sequence. Returns positions and one-/three-letter HGVS changes; "
                "start_position pages long difference lists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {"type": "string"},
                    "sequences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 8,
                    },
                    "start_position": {"type": "integer", "minimum": 1},
                },
                "required": ["reference", "sequences"],
            },
        },
    },
] + online.OFFICIAL_SPECS
_FUNCTIONS = {
    "query_gene_sets": query_gene_sets,
    "query_interactions": query_interactions,
    "query_disease_evidence": query_disease_evidence,
    "query_clinvar_records": query_clinvar_records,
    "query_protein_records": query_protein_records,
    "compare_sequences": compare_sequences,
}


def invoke(name: str, arguments: dict) -> dict:
    if name in online.OFFICIAL_NAMES:
        return online.invoke(name, arguments)
    function = _FUNCTIONS.get(name)
    if function is None:
        return {"status": "unknown_tool", "name": name}
    try:
        return function(**arguments)
    except (TypeError, ValueError) as exc:
        return {"status": "invalid_arguments", "error": str(exc)}
