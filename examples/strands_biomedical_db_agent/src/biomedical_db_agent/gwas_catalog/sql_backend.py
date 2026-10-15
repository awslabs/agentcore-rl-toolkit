"""Bounded read-only SQL with a reusable connection to a verified GWAS snapshot."""

from __future__ import annotations

import datetime
import decimal
import hashlib
import json
import math
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Lock, Timer
from typing import Any

import duckdb

from . import s3_snapshot
from .catalog import DATASETS, RELEASE, catalog_schema, configured_sources, register_catalog
from .hashing import sha256_file

GENERATED = Path(__file__).resolve().parent / "resources"


@dataclass(frozen=True)
class QueryLimits:
    rows: int = 200
    response_bytes: int = 64_000
    query_bytes: int = 32_000
    parameter_bytes: int = 32_000
    timeout_seconds: float = 30
    memory_mb: int = 512
    threads: int = 1

    def __post_init__(self) -> None:
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError("all query limits must be positive")


def failure(code: str, message: str, snapshot: dict) -> dict:
    return {"data": None, "errors": [{"code": code, "message": message}], "snapshot": snapshot}


def encode_value(value: Any, sql_type: Any) -> Any:
    """Preserve nested SQL values and non-string MAP keys without JSON coercion."""
    if value is None:
        return None
    kind = sql_type.id
    if kind in {"interval", "union", "blob", "bit", "timestamp_ns", "time_ns"}:
        raise ValueError(f"{sql_type} requires an explicit SQL cast to VARCHAR for lossless output")
    if kind == "map":
        key_type, value_type = (child[1] for child in sql_type.children)
        # DuckDB uses a dict for hashable keys, key/value arrays for unhashable keys.
        if key_type.id in {"list", "array", "map", "struct"}:
            pairs = zip(value["key"], value["value"], strict=True)
        else:
            pairs = value.items()
        return [[encode_value(key, key_type), encode_value(item, value_type)] for key, item in pairs]
    if kind in {"list", "array"}:
        child_type = sql_type.children[0][1]
        return [encode_value(item, child_type) for item in value]
    if kind == "struct":
        return {name: encode_value(value[name], child_type) for name, child_type in sql_type.children}
    if isinstance(value, decimal.Decimal):
        if not value.is_finite():
            raise ValueError("non-finite DECIMAL; cast explicitly in SQL")
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        # The Python driver maps infinity/out-of-range dates onto extreme sentinels.
        if isinstance(value, datetime.date) and value.year in {1, 9999}:
            raise ValueError("extreme/infinite temporal value; cast explicitly to VARCHAR in SQL")
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"unsupported or non-finite {sql_type}; cast explicitly to VARCHAR in SQL")


def _execute(
    connection: duckdb.DuckDBPyConnection, query: str, parameters: dict, snapshot: dict, limits: QueryLimits
) -> dict:
    try:
        statements = connection.extract_statements(query)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            return failure("READ_ONLY_QUERY_REQUIRED", "Submit one SELECT, WITH, DESCRIBE, or SHOW query.", snapshot)
        if statements[0].named_parameters != set(parameters):
            return failure("INVALID_PARAMETERS", "Named parameters must exactly match the SQL placeholders.", snapshot)
        cursor = connection.execute(query, parameters)
        description = cursor.description
        columns = [{"name": item[0], "type": str(item[1])} for item in description]
        raw_rows = cursor.fetchmany(limits.rows + 1)
        if len(raw_rows) > limits.rows:
            return failure(
                "ROW_LIMIT_EXCEEDED", f"Return at most {limits.rows} rows; narrow the query or add LIMIT.", snapshot
            )
    except duckdb.Error as error:
        return failure("SQL_ERROR", str(error), snapshot)
    try:
        rows = [
            [encode_value(value, column[1]) for value, column in zip(row, description, strict=True)] for row in raw_rows
        ]
        result = {
            "data": {"columns": columns, "rows": rows, "row_count": len(rows)},
            "errors": [],
            "snapshot": snapshot,
        }
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode()) > limits.response_bytes:
            return failure(
                "RESPONSE_LIMIT_EXCEEDED",
                f"Return fewer rows/columns; the response limit is {limits.response_bytes} bytes.",
                snapshot,
            )
        return result
    except (TypeError, ValueError, OverflowError) as error:
        return failure("ENCODING_ERROR", str(error), snapshot)


class SnapshotSQLBackend:
    """Deployment-owned snapshot, SQL catalog and limits; no model-controlled paths."""

    def __init__(self, *, limits: QueryLimits | None = None) -> None:
        self.limits = limits or QueryLimits()
        self.sources = configured_sources()
        self.remote_snapshot = None
        if any(path.startswith("s3://") for path in self.sources.values()):
            self.remote_snapshot = {"sources": self.sources}
        manifest_path = GENERATED / "snapshot.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["release"] != RELEASE or manifest["duckdb_version"] != duckdb.__version__:
            raise ValueError("snapshot release or DuckDB version differs from the pinned manifest")
        self.verify_files()
        self.schema = (GENERATED / "gwas_catalog_schema.sql").read_text()
        if hashlib.sha256(self.schema.encode()).hexdigest() != manifest["schema_sha256"]:
            raise ValueError("SQL schema checksum differs from the snapshot manifest")
        self.snapshot = {
            "release": RELEASE,
            "manifest_sha256": sha256_file(manifest_path),
            "duckdb_version": duckdb.__version__,
        }
        self.file_signatures = None if self.remote_snapshot is not None else self._signatures()
        self._scratch = tempfile.TemporaryDirectory(prefix="gwas-sql-")
        self.database = str(Path(self._scratch.name) / "catalog.duckdb")
        self._lock = Lock()
        self._connection = None
        try:
            with duckdb.connect(self.database, config={"threads": 1, "memory_limit": "512MB"}) as connection:
                s3_snapshot.configure_connection(connection, self.sources)
                register_catalog(connection, self.sources)
                if catalog_schema(connection) != self.schema:
                    raise ValueError("registered SQL schema differs from the pinned prompt schema")
                self.extensions = dict(
                    connection.execute(
                        "SELECT extension_name, extension_version FROM duckdb_extensions() "
                        "WHERE loaded ORDER BY extension_name"
                    ).fetchall()
                )
            self._connection = duckdb.connect(
                self.database,
                read_only=True,
                config={
                    "threads": self.limits.threads,
                    "memory_limit": f"{self.limits.memory_mb}MB",
                    "max_temp_directory_size": "0B",
                    "autoinstall_known_extensions": False,
                    "autoload_known_extensions": False,
                    "allow_community_extensions": False,
                },
            )
            s3_snapshot.configure_connection(self._connection, self.sources)
            self._connection.execute("SET allowed_paths = ?", [list(self.sources.values())])
            self._connection.execute("SET enable_progress_bar = false")
            self._connection.execute("SET enable_logging = false")
            self._connection.execute("SET enable_external_access = false")
            self._connection.execute("SET TimeZone = 'UTC'")
            self._connection.execute("SET lock_configuration = true")
        except Exception:
            self.close()
            raise

    def verify_files(self) -> None:
        if self.remote_snapshot is not None:
            s3_snapshot.verify_objects(self.sources)
            return
        for dataset in DATASETS:
            path = Path(self.sources[dataset.name])
            if path.stat().st_size != dataset.size_bytes or sha256_file(path) != dataset.sha256:
                raise ValueError(f"{dataset.name}: snapshot checksum/size mismatch")

    def _signatures(self) -> dict:
        return {
            name: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            for name, path in self.sources.items()
            for stat in [Path(path).stat()]
        }

    def execute(self, query: str, parameters: dict | None = None) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query.encode()) > self.limits.query_bytes:
            return failure(
                "INVALID_QUERY", f"Provide nonempty SQL within {self.limits.query_bytes} bytes.", self.snapshot
            )
        if parameters is not None and (
            not isinstance(parameters, dict) or any(not isinstance(key, str) for key in parameters)
        ):
            return failure(
                "INVALID_PARAMETERS", "parameters must be an object with string keys or null.", self.snapshot
            )
        try:
            encoded = json.dumps(parameters, allow_nan=False)
            if len(encoded.encode()) > self.limits.parameter_bytes:
                raise ValueError("parameter size limit exceeded")
            if self.file_signatures is not None and self._signatures() != self.file_signatures:
                return failure(
                    "SNAPSHOT_CHANGED", "Snapshot files changed; stop and verify the release.", self.snapshot
                )
        except (TypeError, ValueError, OSError) as error:
            return failure(
                "INVALID_PARAMETERS" if not isinstance(error, OSError) else "SNAPSHOT_ERROR", str(error), self.snapshot
            )
        # A connection executes one query at a time; interruption belongs to that query.
        with self._lock:
            if self._connection is None:
                raise RuntimeError("SQL backend is closed")
            timed_out = Event()

            def interrupt() -> None:
                timed_out.set()
                self._connection.interrupt()

            timer = Timer(self.limits.timeout_seconds, interrupt)
            timer.daemon = True
            timer.start()
            try:
                result = _execute(self._connection, query, parameters or {}, self.snapshot, self.limits)
            finally:
                timer.cancel()
                # Finish any in-flight interrupt before another query can use the connection.
                timer.join()
            if timed_out.is_set():
                return failure(
                    "QUERY_TIMEOUT", f"Query exceeded {self.limits.timeout_seconds:g} seconds.", self.snapshot
                )
            if self.file_signatures is not None and self._signatures() != self.file_signatures:
                return failure(
                    "SNAPSHOT_CHANGED", "Snapshot changed during the query; discard this result.", self.snapshot
                )
            return result

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            self._scratch.cleanup()

    def __enter__(self) -> SnapshotSQLBackend:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
