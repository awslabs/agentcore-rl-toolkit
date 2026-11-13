"""The GWAS tasks' two database tools, as OpenAI function tools.

The executors are the upstream GWAS agent's own modules, vendored unchanged under clients/:
  * query_open_targets_graphql -> clients.opentargets_graphql.query_open_targets_graphql
    (public Open Targets Platform API; read-only check, 512 KB response limit)
  * query_gwas_catalog_sql     -> clients.gwas_catalog_sql.sql_backend.SnapshotSQLBackend.execute
    (DuckDB over the pinned 2026-03-17 parquet files; 200 rows, 64 KB, 30 s per query)
Names, descriptions and parameter schemas are copied from the upstream tool definitions.

A tool exception reaches the model the way the upstream agent framework reports it, as the text
"Error: <message>". Status is recorded for the trace only: `backend_error` marks an unavailable backend
(infrastructure), `error` a rejected call that the model caused.
"""
from __future__ import annotations

import json

from .clients.opentargets_graphql import query_open_targets_graphql as _ot_query

OT_TOOL = {
    "type": "function",
    "function": {
        "name": "query_open_targets_graphql",
        "description": "Execute a raw GraphQL query against Open Targets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "One read-only GraphQL query for the fixed Open Targets endpoint.",
                },
                "variables": {
                    "anyOf": [{"type": "object", "additionalProperties": True}, {"type": "null"}],
                    "default": None,
                    "description": "Optional JSON values for variables declared by the GraphQL query.",
                },
            },
            "required": ["query"],
        },
    },
}

GWAS_TOOL = {
    "type": "function",
    "function": {
        "name": "query_gwas_catalog_sql",
        "description": (
            "Query GWAS Catalog with general SQL, including joins, aggregates and schema discovery.\n\n"
            "Returns data (columns with names/types, positional rows, row_count),\n"
            "errors (code/message), and snapshot (release, manifest_sha256, duckdb_version).\n"
            "Empty results are successful. Query failures or limits return data=null\n"
            "and errors, without silent truncation. Dates/times and decimals are\n"
            "strings; MAP values are arrays of key/value pairs."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1, "description": "One read-only DuckDB SQL query."},
                "parameters": {
                    "anyOf": [{"type": "object", "additionalProperties": True}, {"type": "null"}],
                    "default": None,
                    "description": "Named JSON values bound to $name SQL parameters.",
                },
            },
        },
    },
}

# GWAS backend failure codes that mean the snapshot or worker broke, not that the query was wrong.
GWAS_INFRA_CODES = {"WORKER_ERROR", "SNAPSHOT_CHANGED", "SNAPSHOT_ERROR"}


def ot_invoke(arguments: dict) -> tuple[str, str, object]:
    """Return (status, content the model sees, recorded result)."""
    try:
        result = _ot_query(query=arguments.get("query"), variables=arguments.get("variables"))
    except RuntimeError as exc:  # endpoint unavailable, non-JSON body
        return "backend_error", f"Error: {exc}", {"error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # invalid query, limits, unparsable GraphQL
        return "error", f"Error: {exc}", {"error": f"{type(exc).__name__}: {exc}"}
    return "ok", json.dumps(result, ensure_ascii=False), result


def gwas_invoke(backend, arguments: dict) -> tuple[str, str, object]:
    try:
        result = backend.execute(arguments.get("query"), arguments.get("parameters"))
    except Exception as exc:
        return "backend_error", f"Error: {exc}", {"error": f"{type(exc).__name__}: {exc}"}
    codes = {e.get("code") for e in result.get("errors") or []}
    status = "backend_error" if codes & GWAS_INFRA_CODES else ("error" if codes else "ok")
    return status, json.dumps(result, ensure_ascii=False), result
