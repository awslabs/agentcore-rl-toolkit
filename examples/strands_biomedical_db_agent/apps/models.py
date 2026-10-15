"""Training request shared by the two Runtime entrypoints."""

from pydantic import Field

from biomedical_db_agent.config import ModelConfig
from biomedical_db_agent.contracts import DatabaseRequest


class InvocationRequest(DatabaseRequest):
    expected_answers: list[str] = Field(min_length=1)
    config: ModelConfig = Field(alias="_config")
