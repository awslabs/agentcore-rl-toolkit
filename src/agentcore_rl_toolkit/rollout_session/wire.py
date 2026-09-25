"""The rollout dump an agent server returns for one rollout."""

from typing import Literal

from pydantic import BaseModel, Field


class RolloutDumpResponse(BaseModel):
    response_type: Literal["rollout_dump_response"] = Field(default="rollout_dump_response")
    # Per-rollout scalar metrics; each harness fills in whatever it measures.
    metrics: dict[str, float] = Field(default_factory=dict)
    task_output: dict | None
    reward: float | None
    exception: str | None

    def failure_reason(self) -> str | None:
        """Why this dump does not describe a succesfully finished rollout, or None if it does."""
        if self.exception is not None:
            return self.exception
        if self.reward is None:
            return "Rollout reported no reward and no exception"
        if self.task_output is None:
            return "Rollout reported no task output and no exception"
        return None

    def is_successful(self) -> bool:
        """Whether this dump describes a rollout whose reward may be used."""
        return self.failure_reason() is None
