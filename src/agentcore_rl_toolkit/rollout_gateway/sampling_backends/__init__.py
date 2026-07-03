"""Inference/sampling engines behind the :class:`SamplingBackend` protocol.

Named ``sampling_backends`` (not ``backends``) to avoid colliding with the top-level
``backends/`` package, which holds training-framework integrations. These are the
token-in / token-out inference engines.

The concrete backends (``VllmHttpBackend``, ``SglangHttpBackend``, ``TinkerSdkBackend``)
are imported directly from their modules to avoid pulling optional deps (aiohttp /
tinker) at package import.
"""

from .base import SamplingBackend

__all__ = ["SamplingBackend"]
