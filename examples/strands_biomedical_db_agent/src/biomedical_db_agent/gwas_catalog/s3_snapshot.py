"""Verified immutable S3 releases read through DuckDB's native httpfs extension."""

from __future__ import annotations

import base64
import os
from urllib.parse import urlsplit

from .catalog import DATASETS

EXTENSIONS_ENV = "GWAS_CATALOG_DUCKDB_EXTENSION_DIR"


def object_args(uri: str) -> dict:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
        or parsed.username
        or ":" in parsed.netloc
    ):
        raise ValueError("Expected a plain s3://bucket/key URI")
    return {"Bucket": parsed.netloc, "Key": parsed.path.lstrip("/")}


def s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client("s3", config=Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 2}))


def verify_objects(sources: dict[str, str], *, client=None) -> None:
    client = client or s3_client()
    if set(sources) != {dataset.name for dataset in DATASETS}:
        raise ValueError("Expected exactly the three snapshot datasets")
    for dataset in DATASETS:
        args = object_args(sources[dataset.name])
        head = client.head_object(**args, ChecksumMode="ENABLED")
        expected = base64.b64encode(bytes.fromhex(dataset.sha256)).decode()
        if (
            head.get("ContentLength") != dataset.size_bytes
            or head.get("ChecksumSHA256") != expected
            or head.get("ChecksumType") != "FULL_OBJECT"
        ):
            raise ValueError(f"{dataset.name}: S3 snapshot size/full-object SHA-256 differs")


def configure_connection(connection, sources: dict[str, str]) -> None:
    if not any(uri.startswith("s3://") for uri in sources.values()):
        return
    if directory := os.environ.get(EXTENSIONS_ENV):
        connection.execute("SET extension_directory = ?", [directory])
    # Installed ahead of time, never downloaded by model queries.
    connection.execute("LOAD httpfs")
    connection.execute("LOAD aws")
    connection.execute("CREATE SECRET gwas_s3 (TYPE s3, PROVIDER credential_chain)")
    connection.execute("SET enable_http_metadata_cache = true")
    connection.execute("SET s3_version_id_pinning = true")
