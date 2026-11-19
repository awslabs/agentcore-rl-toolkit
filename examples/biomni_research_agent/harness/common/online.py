"""Direct-query online database tools, shared by dbqa and screen.

The agent supplies an endpoint, NCBI search term, or GraphQL query. No tool
accepts a natural-language prompt or calls an auxiliary model. NCBI requests
rotate over the NCBI_API_KEYS keys, each rate-limited on its own across every
tool-service process.
"""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import os
import re
import threading
import time
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests

from .paths import RATE_DIR

BASES = {
    "query_uniprot": "https://rest.uniprot.org",
    "query_ensembl": "https://rest.ensembl.org",
    "query_monarch": "https://api.monarchinitiative.org/v3/api",
    "query_stringdb": "https://version-12-0.string-db.org/api",
    "query_jaspar": "https://jaspar.elixir.no/api/v1",
    "query_ucsc": "https://api.genome.ucsc.edu",
    "query_remap": "https://remap.univ-amu.fr/api/v1",
    "query_quickgo": "https://www.ebi.ac.uk/QuickGO/services",
}
NCBI_HOST = "eutils.ncbi.nlm.nih.gov"
NCBI_KEYS_ENV = "NCBI_API_KEYS"
# NCBI allows 10 requests/s per key; stay just under it.
NCBI_KEY_INTERVAL = 0.11
_KEY_CYCLE = itertools.count()
_KEY_LOCK = threading.Lock()
EXTRA_HOSTS = {"query_stringdb": {"string-db.org", "www.string-db.org"}}
MAX_ENDPOINT_CHARS = 4000
MAX_REPLY_BYTES = 48_000
REQUEST_TIMEOUT = (5, 30)


@lru_cache(maxsize=1)
def ncbi_keys() -> tuple[str, ...]:
    """Comma-separated keys from NCBI_API_KEYS; fail loudly."""
    keys = tuple(k.strip() for k in os.environ.get(NCBI_KEYS_ENV, "").split(",") if k.strip())
    if not keys or len(set(keys)) != len(keys) or any(not re.fullmatch(r"[0-9A-Za-z]{36}", k) for k in keys):
        raise RuntimeError(f"{NCBI_KEYS_ENV} must hold one or more distinct NCBI API keys")
    return keys


def _redact(value):
    if isinstance(value, str):
        for key in ncbi_keys():
            value = value.replace(key, "<ncbi_api_key>")
        return value
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _limit(host: str, interval: float | None = None) -> None:
    """Coordinate requests across tool-service processes through a lock file per host."""
    RATE_DIR.mkdir(parents=True, exist_ok=True)
    path = RATE_DIR / (re.sub(r"[^a-z0-9]+", "_", host.lower()) + ".stamp")
    if interval is None:
        interval = 0.35 if host == NCBI_HOST else 0.12
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        try:
            previous = float(stream.read().strip() or "0")
        except ValueError:
            previous = 0.0
        delay = max(0.0, previous + interval - time.time())
        if delay:
            time.sleep(delay)
        stream.seek(0)
        stream.truncate()
        stream.write(str(time.time()))
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)


def _request(url: str, source: str, *, params: dict | None = None, json_body: dict | None = None) -> dict:
    if (urlparse(url).hostname or "") != NCBI_HOST:
        return _request_once(url, source, params=params, json_body=json_body)
    keys = ncbi_keys()
    with _KEY_LOCK:
        index = next(_KEY_CYCLE) % len(keys)
    fp = hashlib.sha256(keys[index].encode()).hexdigest()[:12]
    return _redact(
        _request_once(
            url,
            source,
            params={**(params or {}), "api_key": keys[index]},
            json_body=json_body,
            limiter=(f"{NCBI_HOST}_sha256_{fp}", NCBI_KEY_INTERVAL),
        )
    )


def _request_once(
    url: str,
    source: str,
    *,
    params: dict | None = None,
    json_body: dict | None = None,
    limiter: tuple[str, float] | None = None,
) -> dict:
    started = time.monotonic()
    host = urlparse(url).hostname or ""
    try:
        _limit(*(limiter or (host,)))
        with requests.request(
            "POST" if json_body is not None else "GET",
            url,
            params=params,
            json=json_body,
            timeout=REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
            stream=True,
        ) as response:
            chunks = []
            size = 0
            for chunk in response.iter_content(chunk_size=8192):
                size += len(chunk)
                if size > MAX_REPLY_BYTES:
                    return {
                        "status": "result_too_large",
                        "source": source,
                        "backend_source": "online",
                        "http_status": response.status_code,
                        "request_url": response.url,
                        "error": "Online response exceeded 48 KB; narrow the query or page it.",
                    }
                chunks.append(chunk)
            body = b"".join(chunks)
            status = response.status_code
            if status == 404:
                return {
                    "status": "not_found",
                    "source": source,
                    "backend_source": "online",
                    "http_status": status,
                    "request_url": response.url,
                }
            if status >= 500 or status in (408, 429, 401, 403):
                return {
                    "status": "backend_error",
                    "source": source,
                    "backend_source": "online",
                    "http_status": status,
                    "request_url": response.url,
                    "retry_after": response.headers.get("Retry-After"),
                    "error": body.decode(errors="replace")[:500],
                }
            if status >= 400:
                return {
                    "status": "query_error",
                    "source": source,
                    "backend_source": "online",
                    "http_status": status,
                    "request_url": response.url,
                    "error": body.decode(errors="replace")[:500],
                }
            try:
                result = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                result = body.decode(errors="replace")
            if isinstance(result, dict) and result.get("errors") and not result.get("data"):
                return {
                    "status": "query_error",
                    "source": source,
                    "backend_source": "online",
                    "http_status": status,
                    "request_url": response.url,
                    "error": str(result["errors"])[:1000],
                }
            return {
                "status": "ok",
                "source": source,
                "backend_source": "online",
                "http_status": status,
                "request_url": response.url,
                "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds": round(time.monotonic() - started, 3),
                "result": result,
            }
    except (requests.RequestException, OSError) as exc:
        return {
            "status": "backend_error",
            "source": source,
            "backend_source": "online",
            "error": str(exc),
        }


def _endpoint_url(name: str, endpoint: str) -> str | None:
    if not isinstance(endpoint, str) or not 1 <= len(endpoint) <= MAX_ENDPOINT_CHARS:
        return None
    base = BASES[name]
    if endpoint.startswith("https://"):
        parsed = urlparse(endpoint)
        allowed = {urlparse(base).hostname, *EXTRA_HOSTS.get(name, set())}
        if parsed.hostname not in allowed or parsed.username or parsed.password:
            return None
        return endpoint
    if "://" in endpoint or endpoint.startswith("//"):
        return None
    return base.rstrip("/") + "/" + endpoint.lstrip("/")


def direct_endpoint(
    name: str,
    endpoint: str,
    max_results: int | None = None,
    verbose: bool = True,
) -> dict:
    if type(verbose) is not bool:  # noqa: E721
        return {"status": "invalid_verbose", "source": name}
    url = _endpoint_url(name, endpoint)
    if url is None:
        return {"status": "invalid_endpoint", "source": name}
    if max_results is not None:
        if type(max_results) is not int or not 1 <= max_results <= 100:  # noqa: E721
            return {"status": "invalid_max_results", "source": name}
        parsed = urlparse(url)
        parameters = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if not any(k in parameters for k in ("limit", "rows", "size", "page_size")):
            parameters["limit"] = str(max_results)
            url = urlunparse(parsed._replace(query=urlencode(parameters)))
    return _request(url, name)


def ncbi_search(database: str, search_term: str, max_results: int = 3) -> dict:
    if (
        database not in {"clinvar", "snp"}
        or not isinstance(search_term, str)
        or not 1 <= len(search_term.strip()) <= 2000
        or type(max_results) is not int  # noqa: E721
        or not 1 <= max_results <= 20
    ):
        return {"status": "invalid_search_term", "source": database}
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search = _request(
        base + "esearch.fcgi",
        database,
        params={
            "db": database,
            "term": search_term,
            "retmode": "json",
            "retmax": max_results,
        },
    )
    if search["status"] != "ok":
        return search
    ids = search["result"].get("esearchresult", {}).get("idlist", [])
    if not ids:
        return {
            "status": "ok",
            "source": database,
            "search_term": search_term,
            "ids": [],
            "records": [],
            "retrieved_utc": search["retrieved_utc"],
        }
    summary = _request(
        base + "esummary.fcgi",
        database,
        params={"db": database, "id": ",".join(ids), "retmode": "json"},
    )
    if summary["status"] != "ok":
        return summary
    return {
        "status": "ok",
        "source": database,
        "search_term": search_term,
        "ids": ids,
        "records": summary["result"],
        "retrieved_utc": summary["retrieved_utc"],
    }


def gnomad_graphql(query: str, variables: dict | None = None) -> dict:
    if (
        not isinstance(query, str)
        or not 1 <= len(query.strip()) <= 8000
        or variables is not None
        and not isinstance(variables, dict)
    ):
        return {"status": "invalid_query", "source": "gnomAD"}
    return _request(
        "https://gnomad.broadinstitute.org/api",
        "gnomAD",
        json_body={"query": query, "variables": variables or {}},
    )


def endpoint_spec(
    name: str,
    description: str,
    *,
    limited: bool = False,
    routing: str = "online",
) -> dict:
    properties = {
        "endpoint": {"type": "string"},
        "verbose": {"type": "boolean"},
    }
    if limited:
        properties["max_results"] = {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
        }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                description
                + " Supply a direct API endpoint. Backend routing: "
                + routing
                + ". No natural-language prompt is accepted."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["endpoint"],
            },
        },
    }


OFFICIAL_SPECS = [
    endpoint_spec(
        "query_uniprot",
        "Query the online UniProt REST API for proteins and accessions.",
    ),
    endpoint_spec(
        "query_ensembl",
        "Query Ensembl REST routes for genes, variants, and sequences.",
        routing="fixed Ensembl 116 local projection, strict-local without online fallback",
    ),
    endpoint_spec(
        "query_monarch",
        "Query Monarch Initiative API routes for genes, diseases, phenotypes and associations.",
        limited=True,
        routing="local KG entity lookups; other routes use the official online service",
    ),
    {
        "type": "function",
        "function": {
            "name": "query_clinvar",
            "description": (
                "Search the online NCBI ClinVar database with a direct "
                "E-utilities search term. No natural-language prompt is accepted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["search_term"],
            },
        },
    },
    endpoint_spec(
        "query_stringdb",
        "Query the online STRING protein interaction API.",
    ),
    endpoint_spec(
        "query_jaspar",
        "Query the online JASPAR transcription factor profile API.",
    ),
    {
        "type": "function",
        "function": {
            "name": "query_dbsnp",
            "description": ("Search online NCBI dbSNP with a direct E-utilities search term."),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["search_term"],
            },
        },
    },
    endpoint_spec(
        "query_ucsc",
        "Query the online UCSC Genome Browser API for genome regions and annotations.",
    ),
    endpoint_spec(
        "query_remap",
        "Query the online ReMap API for transcription factor binding evidence.",
    ),
    endpoint_spec(
        "query_quickgo",
        "Query the online QuickGO API for Gene Ontology terms and annotations.",
        limited=True,
    ),
    {
        "type": "function",
        "function": {
            "name": "query_gnomad",
            "description": (
                "Execute a model-written GraphQL query against the online "
                "gnomAD API for population variant evidence."
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
OFFICIAL_NAMES = {spec["function"]["name"] for spec in OFFICIAL_SPECS}


def invoke(name: str, arguments: dict) -> dict:
    if name not in OFFICIAL_NAMES:
        return {"status": "unknown_tool", "name": name}
    try:
        if name == "query_clinvar":
            return ncbi_search("clinvar", **arguments)
        if name == "query_dbsnp":
            return ncbi_search("snp", **arguments)
        if name == "query_gnomad":
            return gnomad_graphql(**arguments)
        return direct_endpoint(name, **arguments)
    except (TypeError, ValueError) as exc:
        return {"status": "invalid_arguments", "error": str(exc)}
