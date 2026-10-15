"""Bounded read-only SQL tool for the GWAS Catalog agent."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from strands import tool

from .sql_backend import SnapshotSQLBackend

INPUT_SCHEMA = {
    "json": {
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
}


def make_sql_tool(backend: SnapshotSQLBackend) -> Any:
    @tool(inputSchema=INPUT_SCHEMA)
    async def query_gwas_catalog_sql(
        query: Annotated[str, "One read-only DuckDB query over the configured GWAS Catalog snapshot."],
        parameters: Annotated[
            dict[str, Any] | None, "Optional named JSON values bound to $name SQL parameters."
        ] = None,
    ) -> dict[str, Any]:
        """Query GWAS Catalog with general SQL, including joins, aggregates and schema discovery.

        Returns data (columns with names/types, positional rows, row_count),
        errors (code/message), and snapshot (release, manifest_sha256, duckdb_version).
        Empty results are successful. Query failures or limits return data=null
        and errors, without silent truncation. Dates/times and decimals are
        strings; MAP values are arrays of key/value pairs.
        """
        return await asyncio.to_thread(backend.execute, query, parameters)

    return query_gwas_catalog_sql
