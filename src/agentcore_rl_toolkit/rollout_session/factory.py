"""The single config -> :class:`RolloutSession` mapping, kept verl-free.

``cfg["backend"]`` is the import path of the session class, e.g.
``agentcore_rl_toolkit.rollout_session.agentcore_a2a_session.AgentCoreA2ASession``, and
the class builds itself with a ``from_config(session_id, cfg, session_state)``
classmethod. Nothing here lists the backends, so adding one -- in this package or any
other importable one -- never touches this module. A backend's module is imported only
when a config names it, so its optional dependencies stay optional.

``cfg`` may also carry other backends' keys: each ``from_config`` reads its own by name
and ignores the rest.
"""

import functools
import importlib
from collections.abc import Mapping
from typing import Any

from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict

from .lifecycle import RolloutSession

EXAMPLE_BACKEND = "agentcore_rl_toolkit.rollout_session.agentcore_a2a_session.AgentCoreA2ASession"


def make_session(
    session_id: str,
    cfg: Mapping[str, Any],
    meta: PersistentDict,
) -> RolloutSession:
    """Construct the rollout session whose class ``cfg["backend"]`` names."""
    return resolve_backend(cfg["backend"]).from_config(session_id, cfg, meta)


@functools.cache
def resolve_backend(path: str) -> type:
    """The session class at import path ``path``, which must have ``from_config``."""
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(
            f"rollout_session_backend.backend={path!r} is not an import path. Name the session "
            f"class by its full path, e.g. {EXAMPLE_BACKEND!r}."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ValueError(f"rollout_session_backend.backend={path!r}: cannot import {module_name!r}") from e
    cls = getattr(module, attr, None)
    if cls is None:
        raise ValueError(f"rollout_session_backend.backend={path!r}: {module_name!r} has no {attr!r}")
    if not callable(getattr(cls, "from_config", None)):
        raise ValueError(
            f"rollout_session_backend.backend={path!r} has no from_config(session_id, cfg, session_state) "
            "classmethod, so it cannot be built from config."
        )
    return cls


def require(cfg: Mapping[str, Any], backend: type, *names: str) -> list[Any]:
    """The values of ``names`` in ``cfg``, raising if any is missing or None."""
    missing = [name for name in names if cfg.get(name) is None]
    if missing:
        raise ValueError(f"rollout_session_backend for {backend.__name__} needs {missing}")
    return [cfg[name] for name in names]
