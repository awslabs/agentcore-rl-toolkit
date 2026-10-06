"""The single config -> :class:`RolloutSession` mapping, kept verl-free.

``cfg["backend"]`` is the import path of the session class, e.g.
``agentcore_rl_toolkit.rollout_session.agentcore_a2a_session.AgentCoreA2ASession``, and
the class is constructed as
``backend(session_id, session_state, **backend_config)``. The constructor signature is
therefore the backend's config schema: missing and unknown keys are errors. Nothing here
lists the backends, so adding one -- in this package or any other importable one -- never
touches this module. A backend's module is imported only when a config names it, so its
optional dependencies stay optional.
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
    backend_config = {name: value for name, value in cfg.items() if name != "backend"}
    return resolve_backend(cfg["backend"])(session_id, meta, **backend_config)


@functools.cache
def resolve_backend(path: str) -> type:
    """Return the session class at import path ``path``."""
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
    if not isinstance(cls, type):
        raise ValueError(f"rollout_session_backend.backend={path!r} is not a class")
    return cls
