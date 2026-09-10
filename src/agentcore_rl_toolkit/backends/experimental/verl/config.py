from dataclasses import dataclass
from typing import Any


@dataclass
class RolloutSessionAgentLoopConfig:
    """Typed schema of the ``rollout_session_agent_loop`` config node.

    The yaml node carries ``_target_`` pointing here so the loop can build a typed
    instance via ``hydra.utils.instantiate``.
    """

    # each group is passed whole to its one consumer, not read here
    rollout_gateway: dict[str, Any]
    rollout_session_bounds: dict[str, Any]
    rollout_session_backend: dict[str, Any]

    # per-session sampling defaults for turns the agent does not specify itself;
    # verl's own temperature/top_p/top_k are layered on top per rollout
    rollout_gateway_sampling_params: dict[str, Any]

    # region for every AWS client this loop opens (session table, output bucket, ec2 monitor)
    aws_region: str

    # storage
    dynamodb_table: str | None
    rollout_output_s3: str

    # extra fields folded into every task
    task_kwargs: dict[str, Any]

    # seconds between ec2_monitor polls; <= 0 disables the monitor
    ec2_monitor_poll_interval: float = 30.0
