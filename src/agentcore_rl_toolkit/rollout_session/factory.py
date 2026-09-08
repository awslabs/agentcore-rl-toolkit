"""Config -> :class:`RolloutSession` selection.

The single config->session mapping, so that no caller reimplements (and drifts from)
the selection logic. It lives in the verl-free ``rollout_session`` package so a
session can be constructed straight from a config without standing up verl.
"""

from typing import Protocol, runtime_checkable

from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict

from .agentcore_session import AgentCoreSession
from .docker_session import DockerSession
from .lifecycle import RolloutSession


@runtime_checkable
class SessionConfig(Protocol):
    """The fields :func:`make_session` reads off a config.

    A structural type so this module needs no import of a trainer's config dataclass
    (which would pull in verl). That dataclass satisfies this protocol, and so does
    any lightweight stand-in a test builds.

    ``backend`` keeps its name because it is a *config key* -- hydra's
    ``container_agent_loop.backend``, set in checked-in yaml and on command lines --
    not a Python identifier we are free to rename here.
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
    """Construct the rollout session named by ``cfg.backend``.

    ``cfg`` is anything exposing the :class:`SessionConfig` fields -- a trainer's
    config dataclass, an equivalent ``DictConfig`` node, or a test stand-in.

    The cluster-wide AgentCore session-creation rate limit is not a session's own
    concern: :func:`~.lifecycle.run_rollout_with_bounds` applies it around ``setup``,
    alongside the concurrency semaphores.
    """
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
