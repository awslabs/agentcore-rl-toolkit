import io
import json
from dataclasses import replace

import duckdb
import pytest

from biomedical_db_agent.gwas_catalog import sql_backend
from biomedical_db_agent.gwas_catalog.sql_backend import SnapshotSQLBackend
from biomedical_db_agent.opentargets import graphql_transport as graphql
from biomedical_db_agent.opentargets.schema import load_schema


def test_read_only_snapshot_access(backend, tmp_path):
    assert backend.execute("SELECT count(*) FROM gwas.associations")["data"]["rows"] == [[1]]
    for sql in ["CREATE TABLE x (n INT)", "SELECT 1; SELECT 2"]:
        assert backend.execute(sql)["errors"][0]["code"] == "READ_ONLY_QUERY_REQUIRED"

    path = tmp_path / "outside-snapshot.txt"
    path.write_text("readable fixture")
    query = "SELECT content FROM read_text($path)"
    parameters = {"path": str(path)}
    with duckdb.connect() as connection:
        assert connection.execute(query, parameters).fetchall() == [("readable fixture",)]
    error = backend.execute(query, parameters)["errors"][0]
    assert error["code"] == "SQL_ERROR"
    assert "file system operations are disabled by configuration" in error["message"]


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
    assert backend.execute("SELECT 1")["data"]["rows"] == [[1]]
    source = next(snapshot.glob("*.parquet"))
    source.write_bytes(b"changed")
    assert backend.execute("SELECT 1")["errors"][0]["code"] == "SNAPSHOT_CHANGED"
    with SnapshotSQLBackend() as fresh:
        with pytest.raises(ValueError, match="checksum/size"):
            fresh.execute("SELECT 1")


def test_lazy_initialization_retry_and_reuse(snapshot, monkeypatch):
    register = sql_backend.register_catalog
    calls = 0

    def fail_first_registration(connection, sources):
        nonlocal calls
        calls += 1
        register(connection, sources)
        if calls == 1:
            raise RuntimeError("initialization interrupted")

    monkeypatch.setattr(sql_backend, "register_catalog", fail_first_registration)
    with SnapshotSQLBackend() as backend:
        assert calls == 0
        with pytest.raises(RuntimeError, match="initialization interrupted"):
            backend.execute("SELECT 1")
        for value in (1, 2):
            assert backend.execute("SELECT $value", {"value": value})["data"]["rows"] == [[value]]
        assert calls == 2
    with pytest.raises(RuntimeError, match="closed"):
        backend.execute("SELECT 1")


def test_result_fidelity(backend):
    result = backend.execute("SELECT 1.23456789::DECIMAL(10,8), DATE '2026-03-17', map([1],['a']), [1,NULL]")
    assert result["data"]["rows"] == [["1.23456789", "2026-03-17", [[1, "a"]], [1, None]]]
    json.dumps(result, allow_nan=False)
    result = backend.execute("SELECT $value AS x, $value AS x", {"value": "'; DROP"})
    assert result["data"]["rows"] == [["'; DROP", "'; DROP"]]
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
    with pytest.raises(ValueError, match="only GraphQL query operations"):
        graphql.query_open_targets_graphql(query="mutation X { update }")
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
