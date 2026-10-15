import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from biomedical_db_agent.gwas_catalog.sql_backend import SnapshotSQLBackend
from biomedical_db_agent.opentargets import graphql_transport as graphql
from biomedical_db_agent.opentargets.schema import load_schema


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE x (n INT)",
        "SELECT 1; SELECT 2",
        "SET threads=9",
        "INSTALL httpfs",
        "LOAD httpfs",
        "ATTACH ':memory:' AS x",
        "PRAGMA enable_profiling",
    ],
)
def test_only_read_queries(backend, sql):
    assert backend.execute(sql)["errors"][0]["code"] == "READ_ONLY_QUERY_REQUIRED"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_text('/etc/passwd')",
        "SELECT * FROM read_parquet('https://example.com/data.parquet')",
        "SELECT * FROM query('CREATE TABLE x(n INT)')",
        "SELECT * FROM read_csv('/proc/self/environ')",
    ],
)
def test_external_reads_and_nested_writes_blocked(backend, sql):
    assert backend.execute(sql)["errors"][0]["code"] == "SQL_ERROR"


def test_real_catalog_registration_and_concurrency(backend):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                backend.execute,
                [
                    "SELECT count(*) FROM gwas.associations",
                    "SELECT count(*) FROM gwas.studies",
                ],
            )
        )
    assert [result["data"]["rows"] for result in results] == [[[1]], [[1]]]
    assert backend.execute("SELECT * FROM gwas.samples WHERE false")["data"]["rows"] == []
    result = backend.execute("SELECT $value AS x, $value AS x", {"value": "'; DROP"})
    assert result["data"]["rows"] == [["'; DROP", "'; DROP"]]
    assert backend.execute("SELECT $a", {})["errors"][0]["code"] == "INVALID_PARAMETERS"
    assert backend.execute("SELECT $a", {"a": float("nan")})["errors"][0]["code"] == "INVALID_PARAMETERS"


def test_limits_and_timeout_recovery(backend):
    assert backend.execute("SELECT * FROM range(201)")["errors"][0]["code"] == "ROW_LIMIT_EXCEEDED"
    assert backend.execute("SELECT repeat('x',64000)")["errors"][0]["code"] == "RESPONSE_LIMIT_EXCEEDED"
    backend.limits = replace(backend.limits, timeout_seconds=0.01)
    assert (
        backend.execute("SELECT sum(i * random()) FROM range(1000000000) t(i)")["errors"][0]["code"] == "QUERY_TIMEOUT"
    )
    backend.limits = replace(backend.limits, timeout_seconds=30)
    assert backend.execute("SELECT 1")["data"]["rows"] == [[1]]


def test_snapshot_integrity(snapshot, backend):
    source = next(snapshot.glob("*.parquet"))
    source.write_bytes(b"changed")
    assert backend.execute("SELECT 1")["errors"][0]["code"] == "SNAPSHOT_CHANGED"
    with pytest.raises(ValueError, match="checksum/size"):
        SnapshotSQLBackend()


def test_result_fidelity(backend):
    result = backend.execute("SELECT 1.23456789::DECIMAL(10,8), DATE '2026-03-17', map([1],['a']), [1,NULL]")
    assert result["data"]["rows"] == [["1.23456789", "2026-03-17", [[1, "a"]], [1, None]]]
    json.dumps(result, allow_nan=False)
    for sql in ["SELECT 'NaN'::DOUBLE", "SELECT DATE 'infinity'", "SELECT INTERVAL '1 month'"]:
        assert backend.execute(sql)["errors"][0]["code"] == "ENCODING_ERROR"


class Response(io.BytesIO):
    def __init__(self, status, body):
        super().__init__(json.dumps(body).encode())
        self.status = status


@pytest.fixture
def graphql_endpoint(monkeypatch):
    graphql.verify_release.cache_clear()
    monkeypatch.setenv("OPEN_TARGETS_GRAPHQL_URL", "http://example.test/graphql")
    monkeypatch.setenv("OPEN_TARGETS_EXPECTED_RELEASE", "26.06")

    def install(body, status=200, release="26"):
        def urlopen(request, timeout):
            if "OpenTargetsRelease" in json.loads(request.data)["query"]:
                return Response(200, {"data": {"meta": {"dataVersion": {"year": release, "month": 6}}}})
            return Response(status, body)

        monkeypatch.setattr(graphql.urllib.request, "urlopen", urlopen)

    yield install
    graphql.verify_release.cache_clear()


def test_graphql_validation_and_response_bounds(graphql_endpoint):
    for query in ["mutation X { update }", "query A { a } query B { b }", ""]:
        with pytest.raises(ValueError):
            graphql.query_open_targets_graphql(query=query)
    graphql_endpoint({"data": {"x": "y" * graphql.MAX_RESPONSE_BYTES}})
    with pytest.raises(ValueError, match="response exceeds"):
        graphql.query_open_targets_graphql(query="{ x }")


def test_graphql_failures_are_not_empty_successes(graphql_endpoint):
    graphql_endpoint({"message": "unavailable"}, status=503)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        graphql.query_open_targets_graphql(query="{ meta { name } }")
    graphql_endpoint({"errors": [{"message": "unknown field"}]}, status=400)
    assert graphql.query_open_targets_graphql(query="{ x }")["errors"] == [{"message": "unknown field"}]
    graphql_endpoint({"data": {"target": None}})
    assert graphql.query_open_targets_graphql(query="{ target { id } }") == {"data": {"target": None}, "errors": []}


def test_schema_rejects_release_mismatch(graphql_endpoint):
    graphql_endpoint({"data": {}}, release="25")
    with pytest.raises(RuntimeError, match="release mismatch"):
        load_schema()
    assert graphql._release_from_body({"data": None}) is None
