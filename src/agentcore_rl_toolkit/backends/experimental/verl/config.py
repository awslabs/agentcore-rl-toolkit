from dataclasses import dataclass
from typing import Any


@dataclass
class RolloutSessionAgentLoopConfig:
    """Schema of the ``rollout_session_agent_loop`` config node.

    This dataclass is the single source of truth for the fields the
    :class:`~.rollout_session_agent_loop.RolloutSessionAgentLoop` reads. The yaml
    node carries ``_target_: ...RolloutSessionAgentLoopConfig`` so the loop can
    build a typed instance via ``hydra.utils.instantiate`` instead of poking at
    an untyped :class:`~omegaconf.DictConfig`.

    The three group fields are handed whole to the one component that owns them,
    so a setting added inside a group is a yaml edit plus a change in that
    component -- the loop names only the keys it consumes itself. What each group
    accepts is therefore documented where it is consumed:
    :func:`~agentcore_rl_toolkit.backends.verl.gateway_host.get_or_start_gateway`,
    :func:`~.rollout_session_resources.get_rollout_session_bounds` (with the
    ceilings in :func:`~.rollout_session_resources.start_rollout_session_agent_loop_resources`)
    and :func:`~agentcore_rl_toolkit.rollout_session.factory.make_session`, whose
    ``backend`` key selects the session implementation -- only that backend's
    keys need to be set.
    """

    # groups, each passed through to its one consumer without being read here
    rollout_gateway: dict[str, Any]
    rollout_session_bounds: dict[str, Any]
    rollout_session_backend: dict[str, Any]

    # the gateway's per-session sampling defaults, applied to any turn the agent
    # does not specify itself. Separate from the rollout_gateway group because it
    # is per session rather than per gateway, and so travels through
    # create_session rather than the gateway's constructor. verl's own
    # temperature/top_p/top_k are layered on top of it per rollout.
    rollout_gateway_sampling_params: dict[str, Any]

    # the region every AWS client this loop opens is pinned to: the session-state
    # table, the rollout-output bucket and the ec2 monitor. One key because they
    # are one deployment's resources; the yaml defaults it to $AWS_REGION.
    aws_region: str

    # storage
    dynamodb_table: str | None
    rollout_output_s3: str

    # extra fields folded into every task
    task_kwargs: dict[str, Any]

    # seconds between ec2_monitor polls, which stamp ec2_instance_id onto the
    # sessions of the capacity provider's running instances; <= 0 disables the
    # monitor (see agentcore_rl_toolkit.aws_tools.ec2_monitor)
    ec2_monitor_poll_interval: float = 30.0
