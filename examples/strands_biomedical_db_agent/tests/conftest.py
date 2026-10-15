"""Synthetic database fixtures; no scientific labels or downloaded source rows."""

import hashlib
from dataclasses import replace

import duckdb
import pytest

from biomedical_db_agent.gwas_catalog import catalog, s3_snapshot, sql_backend


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    datasets = []
    for dataset in catalog.DATASETS:
        path = tmp_path / dataset.filename
        columns = ", ".join(
            f"CAST(NULL AS {column.type}) AS {catalog.quote_identifier(column.source)}" for column in dataset.columns
        )
        with duckdb.connect() as con:
            con.execute(f"COPY (SELECT {columns}) TO '{path}' (FORMAT PARQUET)")
        datasets.append(
            replace(
                dataset,
                size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    for module in (catalog, sql_backend, s3_snapshot):
        monkeypatch.setattr(module, "DATASETS", tuple(datasets))
    monkeypatch.setenv(catalog.DATA_ROOT_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def backend(snapshot):
    with sql_backend.SnapshotSQLBackend() as backend:
        yield backend
