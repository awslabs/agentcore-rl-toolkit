"""Read-only GraphQL tool for the Open Targets agent."""

from __future__ import annotations

from typing import Annotated, Any

from strands import tool

from .graphql_transport import query_open_targets_graphql as _query_open_targets_graphql


@tool
def query_open_targets_graphql(
    query: Annotated[
        str,
        "One read-only GraphQL query for the fixed Open Targets endpoint.",
    ],
    variables: Annotated[
        dict[str, Any] | None,
        "Optional JSON values for variables declared by the GraphQL query.",
    ] = None,
) -> dict[str, Any]:
    """Execute a raw GraphQL query against Open Targets."""
    return _query_open_targets_graphql(query=query, variables=variables)
