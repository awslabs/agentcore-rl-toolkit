"""Read-only GraphQL transport for the fixed Open Targets endpoint."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any

from graphql import OperationType, parse
from graphql.language.ast import OperationDefinitionNode

DEFAULT_ENDPOINT = "http://127.0.0.1:8081/api/v4/graphql"
DEFAULT_EXPECTED_RELEASE = "26.06"
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_QUERY_CHARACTERS = 50_000
MAX_VARIABLE_BYTES = 100_000
MAX_RESPONSE_BYTES = 512_000
META_QUERY = """
query OpenTargetsRelease {
  meta {
    dataVersion {
      year
      month
      iteration
    }
  }
}
"""


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
        raise ValueError("GraphQL response exceeds the tool limit; request fewer fields or a smaller page")
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Open Targets returned non-JSON HTTP {status}") from error
    if not isinstance(body, dict):
        raise RuntimeError("Open Targets returned a non-object JSON response")
    return status, body


def _release_from_body(body: dict[str, Any]) -> str | None:
    version = ((body.get("data") or {}).get("meta") or {}).get("dataVersion") or {}
    year = version.get("year")
    month = version.get("month")
    if year is None or month is None:
        return None
    release = f"{int(year) % 100:02d}.{int(month):02d}"
    iteration = version.get("iteration")
    return f"{release}.{iteration}" if iteration else release


@lru_cache(maxsize=8)
def verify_release(
    endpoint: str,
    expected_release: str,
    timeout: float,
) -> str:
    status, body = _post_json(
        endpoint=endpoint,
        query=META_QUERY,
        variables=None,
        timeout=timeout,
    )
    release = _release_from_body(body)
    if status != 200 or release is None:
        raise RuntimeError(f"Open Targets release preflight failed: HTTP {status}, errors={body.get('errors', [])}")
    if release != expected_release:
        raise RuntimeError(f"Open Targets release mismatch: expected {expected_release}, received {release}")
    return release


def query_open_targets_graphql(
    *,
    query: str,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one read-only query against the fixed Open Targets endpoint."""
    _validate_query(query)
    variables = variables or {}
    try:
        variable_bytes = len(json.dumps(variables, separators=(",", ":")).encode())
    except (TypeError, ValueError) as error:
        raise ValueError("variables must be JSON serializable") from error
    if variable_bytes > MAX_VARIABLE_BYTES:
        raise ValueError(f"variables exceed the {MAX_VARIABLE_BYTES}-byte limit")

    endpoint = os.environ.get("OPEN_TARGETS_GRAPHQL_URL", DEFAULT_ENDPOINT)
    expected_release = os.environ.get(
        "OPEN_TARGETS_EXPECTED_RELEASE",
        DEFAULT_EXPECTED_RELEASE,
    )
    timeout = float(
        os.environ.get(
            "OPEN_TARGETS_GRAPHQL_TIMEOUT_SECONDS",
            DEFAULT_TIMEOUT_SECONDS,
        )
    )
    verify_release(endpoint, expected_release, timeout)
    try:
        status, body = _post_json(
            endpoint=endpoint,
            query=query,
            variables=variables,
            timeout=timeout,
        )
    except urllib.error.URLError as error:
        raise RuntimeError(f"Open Targets endpoint is unavailable: {error.reason}") from error
    if status != 200 and not body.get("errors"):
        raise RuntimeError(f"Open Targets query failed with HTTP {status}")
    return {
        "data": body.get("data"),
        "errors": body.get("errors") or [],
    }
