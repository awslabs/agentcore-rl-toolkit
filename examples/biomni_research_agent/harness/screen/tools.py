"""screen's ten tools: ORCS SQL without the hit flag (BioGRID ORCS 2.0.18), four direct-query tools of
common/online.py, and KEGG, Reactome, Open Targets, NCBI Gene and PubMed.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time

from ..common import online
from ..common.paths import DATA

# -------------------------------------------------------------------- ORCS SQL
# Read-only SQL over the fixed BioGRID ORCS 2.0.18 snapshot, hit flag removed.
#
# The scores.hit flag is not in the index, so significance has to be judged from
# the scores and each screen's own criteria.
# The model writes the query; nothing is matched, ranked or selected here.

ORCS = DATA / "screen/orcs_nohit.sqlite"
TABLES = {"screens", "scores"}
SOURCE = "BioGRID ORCS 2.0.18 local snapshot"
MAX_QUERY_CHARS = 8000
MAX_ROWS = 100
MAX_RESULT_BYTES = 48_000
MAX_QUERY_SECONDS = 15.0
# Index-backed lookups need a few thousand VM steps; a full scan of the
# 26M-row scores table cannot finish inside the time limit either way.
MAX_SQL_STEPS = 50_000_000
BACKEND_FAILURES = (
    "unable to open",
    "disk i/o",
    "database is locked",
    "malformed",
    "file is not a database",
    "out of memory",
)


def _read_only_authorizer(action: int, arg1: str | None, arg2: str | None, *_):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and arg1 in TABLES:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        # Extension loading stays disabled on the connection.
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def query_orcs(query: str, offset: int = 0) -> dict:
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= MAX_QUERY_CHARS:
        return {"status": "invalid_query", "source": SOURCE}
    statement = query.strip().removesuffix(";").strip()
    if not re.match(r"(?is)^(SELECT|WITH)\b", statement):
        return {
            "status": "invalid_query",
            "source": SOURCE,
            "error": "Supply one read-only SELECT query.",
        }
    if type(offset) is not int or not 0 <= offset <= 100_000:  # noqa: E721
        return {"status": "invalid_offset", "source": SOURCE}
    started = time.monotonic()
    try:
        con = sqlite3.connect(f"file:{ORCS}?mode=ro&immutable=1", uri=True, timeout=1.0)
    except sqlite3.Error as exc:
        return {"status": "backend_error", "source": SOURCE, "error": str(exc)}
    steps = 0
    stopped = False

    def check_budget() -> int:
        nonlocal steps, stopped
        steps += 1000
        stopped = steps > MAX_SQL_STEPS or time.monotonic() - started > MAX_QUERY_SECONDS
        return int(stopped)

    try:
        con.execute("PRAGMA query_only=ON")
        con.set_authorizer(_read_only_authorizer)
        con.set_progress_handler(check_budget, 1000)
        # Wrapping gives every query the same explicit pagination contract.
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
                "source": SOURCE,
                "status": "ok",
            }
            if len(json.dumps(prospective, ensure_ascii=False).encode()) > MAX_RESULT_BYTES:
                if not rows:
                    return {
                        "status": "result_too_large",
                        "source": SOURCE,
                        "error": "One row exceeds the result budget; select fewer columns.",
                    }
                break
            rows.append(row)
        return {
            "status": "ok",
            "source": SOURCE,
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
        elif any(marker in message.lower() for marker in BACKEND_FAILURES):
            status = "backend_error"
        else:
            # Everything else (unknown names, syntax, denied tables, overflow)
            # is caused by the submitted query, not by the snapshot.
            status = "query_error"
        return {"status": status, "source": SOURCE, "error": message}
    finally:
        con.close()


ORCS_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "query_orcs",
            "description": (
                "BioGRID ORCS 2.0.18 CRISPR screen database, fixed local snapshot. "
                "Table screens has one row per published screen with the native "
                "ORCS metadata columns screen_id, source_id, source_type, author, "
                "screen_name, scores_size, full_size, full_size_available, "
                "number_of_hits, analysis, significance_indicator, "
                "significance_criteria, throughput, screen_type, screen_format, "
                "experimental_setup, duration, condition_name, condition_dosage, "
                "moi, library, library_type, library_methodology, enzyme, "
                "cell_line, cell_type, phenotype, score_col_count, "
                "score_1_type ... score_5_type, organism_id, organism_official, "
                "notes, source, screen_rationale. Table scores has one row per "
                "gene record reported by a screen: screen_id, symbol, aliases, "
                "score_1 ... score_5. All values are text; "
                "scores.symbol is stored upper-case. Indexes: screens(screen_id), "
                "scores(screen_id, symbol), scores(symbol). A gene with no row in "
                "a screen was not reported by that screen. Write one SQLite "
                "SELECT (WITH is allowed). Results include source, status and "
                "pagination; use offset to continue."
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
    },
]
_ORCS_FUNCTIONS = {"query_orcs": query_orcs}


def orcs_invoke(name: str, arguments: dict) -> dict:
    function = _ORCS_FUNCTIONS.get(name)
    if function is None:
        return {"status": "unknown_tool", "name": name}
    try:
        return function(**arguments)
    except (TypeError, ValueError) as exc:
        return {"status": "invalid_arguments", "error": str(exc)}


# ----------------------------------------------------------- more online tools
# More direct-query online tools for screen (KEGG, Reactome, Open Targets, NCBI Gene and PubMed).
#
# Same contract as common/online.py: the agent supplies an endpoint, an NCBI
# search term or a GraphQL query; nothing accepts a natural-language prompt or
# calls an auxiliary model. All of these are live online services.

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BASES = {
    "query_kegg": "https://rest.kegg.jp",
    "query_reactome": "https://reactome.org/ContentService",
}
OPENTARGETS = "https://api.platform.opentargets.org/api/v4/graphql"


def _endpoint(name: str, endpoint: str) -> dict:
    if not isinstance(endpoint, str) or not 1 <= len(endpoint) <= online.MAX_ENDPOINT_CHARS:
        return {"status": "invalid_endpoint", "source": name}
    if "://" in endpoint or endpoint.startswith("//"):
        base_host = online.urlparse(BASES[name]).hostname
        parsed = online.urlparse(endpoint)
        if parsed.scheme != "https" or parsed.hostname != base_host or parsed.username:
            return {"status": "invalid_endpoint", "source": name}
        url = endpoint
    else:
        url = BASES[name].rstrip("/") + "/" + endpoint.lstrip("/")
    return online._request(url, name)


def query_kegg(endpoint: str) -> dict:
    return _endpoint("query_kegg", endpoint)


def query_reactome(endpoint: str) -> dict:
    return _endpoint("query_reactome", endpoint)


def query_opentarget(query: str, variables: dict | None = None) -> dict:
    if (
        not isinstance(query, str)
        or not 1 <= len(query.strip()) <= 8000
        or variables is not None
        and not isinstance(variables, dict)
    ):
        return {"status": "invalid_query", "source": "query_opentarget"}
    return online._request(
        OPENTARGETS,
        "query_opentarget",
        json_body={"query": query, "variables": variables or {}},
    )


def _search(database: str, source: str, search_term: str, max_results: int) -> dict | list:
    if (
        not isinstance(search_term, str)
        or not 1 <= len(search_term.strip()) <= 2000
        or type(max_results) is not int  # noqa: E721
        or not 1 <= max_results <= 20
    ):
        return {"status": "invalid_search_term", "source": source}
    search = online._request(
        NCBI + "esearch.fcgi",
        source,
        params={"db": database, "term": search_term, "retmode": "json", "retmax": max_results},
    )
    if search["status"] != "ok":
        return search
    return search["result"].get("esearchresult", {}).get("idlist", [])


def query_pubmed(search_term: str, max_results: int = 5) -> dict:
    ids = _search("pubmed", "query_pubmed", search_term, max_results)
    if isinstance(ids, dict):
        return ids
    if not ids:
        return {"status": "ok", "source": "query_pubmed", "search_term": search_term, "ids": [], "abstracts": ""}
    fetched = online._request(
        NCBI + "efetch.fcgi",
        "query_pubmed",
        params={"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "text"},
    )
    if fetched["status"] != "ok":
        return fetched
    return {
        "status": "ok",
        "source": "query_pubmed",
        "search_term": search_term,
        "ids": ids,
        "abstracts": fetched["result"],
        "retrieved_utc": fetched["retrieved_utc"],
    }


def query_geo(search_term: str, max_results: int = 5) -> dict:
    ids = _search("gds", "query_geo", search_term, max_results)
    if isinstance(ids, dict):
        return ids
    if not ids:
        return {"status": "ok", "source": "query_geo", "search_term": search_term, "ids": [], "records": {}}
    summary = online._request(
        NCBI + "esummary.fcgi",
        "query_geo",
        params={"db": "gds", "id": ",".join(ids), "retmode": "json"},
    )
    if summary["status"] != "ok":
        return summary
    return {
        "status": "ok",
        "source": "query_geo",
        "search_term": search_term,
        "ids": ids,
        "records": summary["result"],
        "retrieved_utc": summary["retrieved_utc"],
    }


def _ncbi_spec(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description + " No natural-language prompt is accepted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["search_term"],
            },
        },
    }


def _plain_endpoint_spec(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                description + " Supply a direct API endpoint. Backend routing: "
                "online. No natural-language prompt is accepted."
            ),
            "parameters": {
                "type": "object",
                "properties": {"endpoint": {"type": "string"}},
                "required": ["endpoint"],
            },
        },
    }


EXTRA_SPECS = [
    _ncbi_spec(
        "query_pubmed",
        "Search online NCBI PubMed with a direct E-utilities search term; " "returns PMIDs and abstract text.",
    ),
    _ncbi_spec(
        "query_geo",
        "Search online NCBI GEO DataSets (gds) with a direct E-utilities " "search term; returns record summaries.",
    ),
    _plain_endpoint_spec(
        "query_kegg",
        "Query the online KEGG REST API (rest.kegg.jp) for pathways, genes and modules.",
    ),
    _plain_endpoint_spec(
        "query_reactome",
        "Query the online Reactome Content Service for pathways and reactions.",
    ),
    {
        "type": "function",
        "function": {
            "name": "query_opentarget",
            "description": (
                "Execute a model-written GraphQL query against the online Open "
                "Targets Platform API for targets, diseases and evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "variables": {"type": "object"},
                },
                "required": ["query"],
            },
        },
    },
]
_EXTRA_FUNCTIONS = {
    "query_pubmed": query_pubmed,
    "query_geo": query_geo,
    "query_kegg": query_kegg,
    "query_reactome": query_reactome,
    "query_opentarget": query_opentarget,
}
EXTRA_NAMES = set(_EXTRA_FUNCTIONS)


def extra_invoke(name: str, arguments: dict) -> dict:
    function = _EXTRA_FUNCTIONS.get(name)
    if function is None:
        return {"status": "unknown_tool", "name": name}
    try:
        return function(**arguments)
    except (TypeError, ValueError) as exc:
        return {"status": "invalid_arguments", "error": str(exc)}


# --------------------------------------------------------- tool specs and dispatch

OFFICIAL_FROM_MIXED = ("query_ensembl", "query_uniprot", "query_stringdb", "query_quickgo")
TOOL_SPECS = [
    *ORCS_SPECS,
    *[s for s in online.OFFICIAL_SPECS if s["function"]["name"] in OFFICIAL_FROM_MIXED],
    *EXTRA_SPECS,
]
NAMES = [spec["function"]["name"] for spec in TOOL_SPECS]
if len(NAMES) != len(set(NAMES)) or len(NAMES) != 10:
    raise RuntimeError(f"Unexpected screen mixed tool pool: {NAMES}")


def invoke(name: str, arguments: dict) -> dict:
    if name == "query_orcs":
        return orcs_invoke(name, arguments)
    if name in OFFICIAL_FROM_MIXED:
        return online.invoke(name, arguments)
    if name in EXTRA_NAMES:
        return extra_invoke(name, arguments)
    return {"status": "unknown_tool", "name": name}
