"""``agentcore_rollout_gateway`` — a standalone, backend-agnostic rollout capture layer.

The gateway owns tokenization, renders canonical chat messages to ``token_ids``, calls
a token-in/token-out :class:`SamplingBackend`, and linearizes per-session message trees
into torch-free :class:`TraceRecord` objects for any training backend.

Import layering:
- Everything exported directly here (``TraceRecord``, ``TrajectoryManager``, the
  ``Renderer`` protocol + ``HfTemplateRenderer``, the ``SamplingBackend`` protocol)
  imports with nothing beyond the stdlib — torch-free and aiohttp-free.
  (``HfTemplateRenderer`` needs a HF tokenizer only when *used*, not to import; sglang
  parsers and tinker are lazy/optional.)
- ``RolloutGateway``, ``ThreadedGatewayServer``, and the HTTP adapters require
  ``aiohttp`` (the ``[gateway]`` extra), so both are exposed lazily via
  ``__getattr__`` — importing this package never requires aiohttp.
"""

from .linear import LinearHealer
from .render import HfTemplateRenderer, ParsedOutput, Renderer
from .sampling_backends.base import SamplingBackend
from .trace import BaseTrace, Status, TraceRecord
from .trajectory import MessageNode, TrajectoryManager, TurnRecord

# RolloutGateway is aiohttp-dependent, so it is NOT imported at module top — it is
# exposed lazily via __getattr__ below to keep `import ...rollout_gateway` aiohttp-free.

__all__ = [
    "BaseTrace",
    "HfTemplateRenderer",
    "LinearHealer",
    "MessageNode",
    "ParsedOutput",
    "Renderer",
    "RolloutGateway",
    "SamplingBackend",
    "Status",
    "ThreadedGatewayServer",
    "TraceRecord",
    "TrajectoryManager",
    "TurnRecord",
]


def __getattr__(name: str):
    # These pull in aiohttp; keep them off the plain import path.
    if name == "RolloutGateway":
        from .gateway import RolloutGateway

        return RolloutGateway
    if name == "ThreadedGatewayServer":
        from .server import ThreadedGatewayServer

        return ThreadedGatewayServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
