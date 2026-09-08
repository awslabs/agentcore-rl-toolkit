"""The request/response types spoken over the agent server's HTTP endpoint.

The agent server in the container is invoked by POSTing an :class:`InvocationRequest` to
``/invocations`` and reading back an :class:`InvocationResponse`; a rollout is
four such calls -- setup, start, status (polled), dump. These types *are* that
protocol, so they are the one thing both sides of the container boundary must
agree on.

They live on the caller's side deliberately. The protocol is what the trainer
requires of any agent it can drive, not something each agent gets to define: this
module and the sessions beside it define the contract, and a harness implements it.

Nothing here imports anything but pydantic, and the package around it has no
imports at all at package level, which is what lets a task container install this
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
    # Generic per-rollout scalar metrics keyed by name. Each harness fills in
    # whatever it measures (e.g. llm_latency_sum, tool_calls_time_s, num_tool_calls).
    metrics: dict[str, float] = Field(default_factory=dict)
    task_output: dict | None
    reward: float | None
    exception: str | None

    def failure_reason(self) -> str | None:
        """Why this dump does not describe a usable rollout, or None if it does.

        Which fields mean "failed" is a property of this response, not of any
        harness: while the todo above is outstanding a dump can come back describing
        a rollout that did not work, and every caller would otherwise have to know
        the same three fields to notice. The failure happened inside the container,
        so the only stack trace worth having is the one in ``exception`` -- when it
        is set it is reported verbatim, and the other two branches stand in for a
        container that failed without saying so.

        When the response type is fixed so that a dump implies success, this and
        :meth:`is_successful` go away together.
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
