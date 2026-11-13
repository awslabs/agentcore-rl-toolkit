"""More direct-query online tools for screen (KEGG, Reactome, Open Targets, NCBI Gene and PubMed).

Same contract as common/online.py: the agent supplies an endpoint, an NCBI
search term or a GraphQL query; nothing accepts a natural-language prompt or
calls an auxiliary model. All of these are live online services.
"""

from __future__ import annotations

from ..common import online

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
_FUNCTIONS = {
    "query_pubmed": query_pubmed,
    "query_geo": query_geo,
    "query_kegg": query_kegg,
    "query_reactome": query_reactome,
    "query_opentarget": query_opentarget,
}
EXTRA_NAMES = set(_FUNCTIONS)


def invoke(name: str, arguments: dict) -> dict:
    function = _FUNCTIONS.get(name)
    if function is None:
        return {"status": "unknown_tool", "name": name}
    try:
        return function(**arguments)
    except (TypeError, ValueError) as exc:
        return {"status": "invalid_arguments", "error": str(exc)}
