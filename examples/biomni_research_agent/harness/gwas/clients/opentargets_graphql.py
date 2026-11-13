"""Read-only GraphQL transport for the public Open Targets Platform API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from graphql import OperationType, parse
from graphql.language.ast import OperationDefinitionNode

ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_QUERY_CHARACTERS = 50_000
MAX_VARIABLE_BYTES = 100_000
MAX_RESPONSE_BYTES = 512_000


def _validate_query(query: str) -> None:
    if not query.strip():
        raise ValueError("query cannot be empty")
    if len(query) > MAX_QUERY_CHARACTERS:
        raise ValueError(f"query exceeds the {MAX_QUERY_CHARACTERS}-character limit")
    document = parse(query)
    operations = [definition for definition in document.definitions if isinstance(definition, OperationDefinitionNode)]
    if len(operations) != 1:
        raise ValueError("exactly one GraphQL operation is required")
    if operations[0].operation is not OperationType.QUERY:
        raise ValueError("only GraphQL query operations are allowed")


def _post_json(
    *,
    endpoint: str,
    query: str,
    variables: dict[str, Any] | None,
    timeout: float,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
) -> tuple[int, dict[str, Any]]:
    payload = {
        "query": query,
        "variables": variables or {},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={
            "accept": "application/json",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    try:
        raw = response.read(max_response_bytes + 1)
        status = int(response.status)
    finally:
        response.close()
    if len(raw) > max_response_bytes:
        raise ValueError("GraphQL response exceeds the tool limit; request fewer fields " "or a smaller page")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Open Targets returned non-JSON HTTP {status}") from error
    if not isinstance(body, dict):
        raise RuntimeError("Open Targets returned a non-object JSON response")
    return status, body


def query_open_targets_graphql(
    *,
    query: str,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one read-only query against the public Open Targets endpoint."""
    _validate_query(query)
    variables = variables or {}
    try:
        variable_bytes = len(json.dumps(variables, separators=(",", ":")).encode())
    except (TypeError, ValueError) as error:
        raise ValueError("variables must be JSON serializable") from error
    if variable_bytes > MAX_VARIABLE_BYTES:
        raise ValueError(f"variables exceed the {MAX_VARIABLE_BYTES}-byte limit")

    try:
        _, body = _post_json(
            endpoint=ENDPOINT,
            query=query,
            variables=variables,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except urllib.error.URLError as error:
        raise RuntimeError(f"Open Targets endpoint is unavailable: {error.reason}") from error
    return {
        "data": body.get("data"),
        "errors": body.get("errors") or [],
    }
