"""Open Targets evidence retrieval using the complete GraphQL schema."""

from pathlib import Path

from ..agent import DatabaseAgent
from .schema import load_schema
from .tools import query_open_targets_graphql


def create_agent() -> DatabaseAgent:
    prompt = Path(__file__).with_name("prompt.md").read_text()
    return DatabaseAgent(
        system_prompt=prompt.format(schema=load_schema().rstrip()),
        tools=[query_open_targets_graphql],
    )
