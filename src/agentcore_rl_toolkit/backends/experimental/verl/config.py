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

    # What the loop does when a rollout fails with no trainable tokens.
    # Which of these trains best is an open question -- the knob exists to
    # gather comparable runs, not because one is known to win.
    # ``raise``     -- (default) verl marks the whole prompt group ``failure``. Sync trains the
    #                 surviving siblings; the async trainers evict and refill the entire group.
    # ``empty``     -- return no rows. The group stays ``finished`` and trains its survivors in
    #                 every trainer mode, but the failure leaves no ``agent_loop/*`` metrics.
    on_rollout_failure: str = "raise"

    # Names the agent's own ``metrics`` entries that ride along in verl's
    # ``reward_extra_info``, with the value to report when a rollout does not produce one.
    # Declared rather than discovered because verl reduces this dict across the batch and a
    # key that only some rollouts carry is silently averaged over the wrong denominator; a
    # declared default keeps every row's key set identical. ``reward`` is ignored (verl
    # derives it from ``rm_scores`` itself); ``reward_score`` and ``num_trace_records`` are
    # always added.
    reward_extra_info_defaults: dict[str, float] | None = None
