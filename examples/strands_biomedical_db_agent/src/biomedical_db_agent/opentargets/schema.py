"""Load the schema from the same release-checked endpoint used by the tool."""

import os

from graphql import build_client_schema, get_introspection_query, print_schema

from .graphql_transport import (
    DEFAULT_ENDPOINT,
    DEFAULT_EXPECTED_RELEASE,
    DEFAULT_TIMEOUT_SECONDS,
    _post_json,
    verify_release,
)


def load_schema() -> str:
    endpoint = os.getenv("OPEN_TARGETS_GRAPHQL_URL", DEFAULT_ENDPOINT)
    release = os.getenv("OPEN_TARGETS_EXPECTED_RELEASE", DEFAULT_EXPECTED_RELEASE)
    timeout = float(os.getenv("OPEN_TARGETS_GRAPHQL_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    verify_release(endpoint, release, timeout)
    status, body = _post_json(
        endpoint=endpoint,
        query=get_introspection_query(descriptions=True),
        variables=None,
        timeout=timeout,
        max_response_bytes=5_000_000,
    )
    if status != 200 or body.get("errors") or not body.get("data"):
        raise RuntimeError(f"Open Targets schema request failed: HTTP {status}, errors={body.get('errors', [])}")
    return print_schema(build_client_schema(body["data"])) + "\n"
