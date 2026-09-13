"""The single config -> :class:`RolloutSession` mapping, kept verl-free."""

from typing import Required, TypedDict

from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict

from .agentcore_http_session import AgentCoreHttpSession
from .agentcore_s3_session import AgentCoreS3Session, get_or_create_rollout_client
from .docker_session import DockerSession
from .lifecycle import RolloutSession


class SessionBackendConfig(TypedDict, total=False):
    """The keys make_session reads, as a plain mapping so this module need not import
    the trainer's config dataclass (which would pull in verl).

    Each key belongs to one backend and only that backend's need setting, except
    ``agentcore_runtime_arn``, which both ACR-backed backends read.
    """

    backend: Required[str]
    agentcore_runtime_arn: str | None
    capacity_provider_arn: str | None
    agent_image_uri: str | None
    docker_iam_role_arn: str | None
    docker_log_group: str | None
    docker_log_region: str | None
    # agentcore_s3: where the agent writes its results and under what prefix
    # (`{experiment_name}/{task_id}/{session_id}.json`), plus the boto3 pool size shared by
    # every session of the process, since the client backing them is. There is deliberately
    # no TPS knob: bounds.session_rate_limiter owns ACR throttling.
    rollout_output_s3: str | None
    experiment_name: str | None
    max_pool_connections: int | None


def make_session(
    session_id: str,
    cfg: SessionBackendConfig,
    meta: PersistentDict,
) -> RolloutSession:
    """Construct the rollout session named by ``cfg["backend"]``."""
    kind = cfg["backend"]
    if kind == "agentcore_http":
        runtime_arn = cfg.get("agentcore_runtime_arn")
        capacity_provider_arn = cfg.get("capacity_provider_arn")
        assert runtime_arn is not None and capacity_provider_arn is not None
        return AgentCoreHttpSession(
            session_id,
            session_state=meta,
            runtime_arn=runtime_arn,
            capacity_provider_arn=capacity_provider_arn,
        )
    if kind == "docker":
        agent_image_uri = cfg.get("agent_image_uri")
        iam_role_arn = cfg.get("docker_iam_role_arn")
        log_group = cfg.get("docker_log_group")
        log_region = cfg.get("docker_log_region")
        assert agent_image_uri is not None and iam_role_arn is not None
        assert log_group is not None and log_region is not None
        return DockerSession(
            session_id,
            session_state=meta,
            agent_image_uri=agent_image_uri,
            iam_role_arn=iam_role_arn,
            log_group=log_group,
            log_region=log_region,
        )
    if kind == "agentcore_s3":
        runtime_arn = cfg.get("agentcore_runtime_arn")
        rollout_output_s3 = cfg.get("rollout_output_s3")
        experiment_name = cfg.get("experiment_name")
        assert runtime_arn is not None and rollout_output_s3 is not None and experiment_name is not None
        client = get_or_create_rollout_client(
            agentcore_runtime_arn=runtime_arn,
            rollout_output_s3=rollout_output_s3,
            experiment_name=experiment_name,
            # RolloutClient's own default of 10 queues rollout bursts on the connection
            # pool, so start where the AgentCore agent loop does; raise it toward
            # rollout_concurrency.
            max_pool_connections=cfg.get("max_pool_connections") or 100,
        )
        return AgentCoreS3Session(session_id, session_state=meta, client=client)
    raise ValueError(
        f"Unknown rollout_session_backend.backend={kind!r}, expected 'agentcore_http', 'agentcore_s3' or 'docker'"
    )
