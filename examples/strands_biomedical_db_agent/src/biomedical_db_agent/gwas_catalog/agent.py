"""GWAS Catalog evidence retrieval over the pinned SQL snapshot."""

from pathlib import Path

from ..agent import DatabaseAgent
from .sql_backend import SnapshotSQLBackend
from .tools import make_sql_tool


def create_agent(backend: SnapshotSQLBackend) -> DatabaseAgent:
    prompt = Path(__file__).with_name("prompt.md").read_text()
    return DatabaseAgent(system_prompt=prompt.format(schema=backend.schema), tools=[make_sql_tool(backend)])
