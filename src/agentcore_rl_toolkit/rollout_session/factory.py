"""The single config -> :class:`RolloutSession` mapping, kept verl-free."""

from typing import Protocol, runtime_checkable

from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict

from .agentcore_session import AgentCoreSession
from .docker_session import DockerSession
from .lifecycle import RolloutSession


@runtime_checkable
class SessionConfig(Protocol):
    """The fields :func:`make_session` reads off a config.

    Structural so this module need not import the trainer's config dataclass (which
    would pull in verl). ``backend`` is the hydra key ``container_agent_loop.backend``.
    """

    backend: str
    agentcore_runtime_arn: str | None
    capacity_provider_arn: str | None
    agent_image_uri: str | None
    docker_iam_role_arn: str | None
    docker_log_group: str | None
    docker_log_region: str | None


def make_session(
    session_id: str,
    cfg: SessionConfig,
    meta: PersistentDict,
) -> RolloutSession:
    """Construct the rollout session named by ``cfg.backend``."""
    kind = cfg.backend
    if kind == "agentcore":
        assert cfg.agentcore_runtime_arn is not None and cfg.capacity_provider_arn is not None
        return AgentCoreSession(
            session_id,
            session_state=meta,
            runtime_arn=cfg.agentcore_runtime_arn,
            capacity_provider_arn=cfg.capacity_provider_arn,
        )
    if kind == "docker":
        assert cfg.agent_image_uri is not None and cfg.docker_iam_role_arn is not None
        assert cfg.docker_log_group is not None and cfg.docker_log_region is not None
        return DockerSession(
            session_id,
            session_state=meta,
            agent_image_uri=cfg.agent_image_uri,
            iam_role_arn=cfg.docker_iam_role_arn,
            log_group=cfg.docker_log_group,
            log_region=cfg.docker_log_region,
        )
    raise ValueError(f"Unknown container_agent_loop.backend={kind!r}, expected 'agentcore' or 'docker'")
