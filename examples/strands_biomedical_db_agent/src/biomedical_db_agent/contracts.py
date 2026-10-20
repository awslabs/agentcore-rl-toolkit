"""Request and response shared by both database agents."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AgentName = Literal["opentargets", "gwas_catalog"]


class ResponseStatus(StrEnum):
    """Outcome of a database-subagent request."""

    COMPLETED = "completed"
    UNRESOLVED = "unresolved"
    UNSUPPORTED = "unsupported"


class DatabaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1, description="Self-contained database task, including candidates and evidence scope."
    )

    @field_validator("query")
    @classmethod
    def nonempty_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain text")
        return value


class DatabaseResponse(BaseModel):
    """Minimal decision returned to the calling agent."""

    model_config = ConfigDict(extra="forbid")

    status: ResponseStatus = Field(
        description="Outcome of the database retrieval request.",
    )
    answer: str | None = Field(
        description=(
            "Concise final answer only. For candidate selection, include exactly "
            "one supplied candidate and no explanatory prose; use null when the "
            "status is unresolved or unsupported."
        ),
    )
    decision_summary: str = Field(
        min_length=1,
        description=(
            "Concise, evidence-grounded explanation of why the request was completed, "
            "remains unresolved, or is unsupported."
        ),
    )

    @model_validator(mode="after")
    def validate_outcome(self) -> "DatabaseResponse":
        """Keep status-specific response requirements explicit."""
        if not self.decision_summary.strip():
            raise ValueError("decision_summary must contain a concise explanation")
        if self.status is ResponseStatus.COMPLETED:
            if not self.answer or not self.answer.strip():
                raise ValueError("completed responses require an answer")
        elif self.answer is not None:
            raise ValueError("unresolved and unsupported responses require answer=null")
        return self
