"""The request/response types spoken over the agent server's ``/invocations`` endpoint.

A rollout is four POSTs of an :class:`InvocationRequest` -- setup, start, status
(polled), dump. Pydantic is the only import, so a task container can install this
distribution with ``--no-deps`` and still get the types.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RolloutSetupRequest(BaseModel):
    request_type: Literal["rollout_setup_request"] = Field(default="rollout_setup_request")
    task_input: dict


class RolloutStartRequest(BaseModel):
    request_type: Literal["rollout_start_request"] = Field(default="rollout_start_request")
    rollout_id: str
    task_input: dict


class RolloutStatusRequest(BaseModel):
    request_type: Literal["rollout_status_request"] = Field(default="rollout_status_request")


class RolloutDumpRequest(BaseModel):
    request_type: Literal["rollout_dump_request"] = Field(default="rollout_dump_request")


InvocationInput = RolloutSetupRequest | RolloutStartRequest | RolloutStatusRequest | RolloutDumpRequest


class InvocationRequest(BaseModel):
    payload: InvocationInput = Field(discriminator="request_type")


class RolloutSetupResponse(BaseModel):
    response_type: Literal["rollout_setup_response"] = Field(default="rollout_setup_response")


class RolloutStartResponse(BaseModel):
    response_type: Literal["rollout_start_response"] = Field(default="rollout_start_response")


class RolloutStatusResponse(BaseModel):
    response_type: Literal["rollout_status_response"] = Field(default="rollout_status_response")
    done: bool
    exception: str | None


class RolloutDumpResponse(BaseModel):
    response_type: Literal["rollout_dump_response"] = Field(default="rollout_dump_response")
    # Per-rollout scalar metrics; each harness fills in whatever it measures.
    metrics: dict[str, float] = Field(default_factory=dict)
    task_output: dict | None
    reward: float | None
    exception: str | None

    def failure_reason(self) -> str | None:
        """Why this dump does not describe a usable rollout, or None if it does.

        A dump can come back describing a rollout that did not work; the two
        non-``exception`` branches stand in for a container that failed without saying
        so. Goes away once a dump implies success.
        """
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


InvocationOutput = RolloutSetupResponse | RolloutStartResponse | RolloutStatusResponse | RolloutDumpResponse


class InvocationResponse(BaseModel):
    payload: InvocationOutput = Field(discriminator="response_type")
