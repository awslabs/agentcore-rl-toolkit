"""The single config -> :class:`RolloutSession` mapping, kept verl-free."""

from typing import Required, TypedDict

from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict

from .agentcore_session import AgentCoreSession
from .docker_session import DockerSession
from .lifecycle import RolloutSession


class SessionBackendConfig(TypedDict, total=False):
    """The keys make_session reads, as a plain mapping so this module need not import
    the trainer's config dataclass (which would pull in verl).

    Every key other than ``backend`` belongs to one backend; only that backend's need setting.
    """

    backend: Required[str]
    agentcore_runtime_arn: str | None
    capacity_provider_arn: str | None
    agent_image_uri: str | None
    docker_iam_role_arn: str | None
    docker_log_group: str | None
    docker_log_region: str | None


def make_session(
    session_id: str,
    cfg: SessionBackendConfig,
    meta: PersistentDict,
) -> RolloutSession:
    """Construct the rollout session named by ``cfg["backend"]``."""
    kind = cfg["backend"]
    if kind == "agentcore":
        runtime_arn = cfg.get("agentcore_runtime_arn")
        capacity_provider_arn = cfg.get("capacity_provider_arn")
        assert runtime_arn is not None and capacity_provider_arn is not None
        return AgentCoreSession(
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
    raise ValueError(f"Unknown rollout_session_backend.backend={kind!r}, expected 'agentcore' or 'docker'")
