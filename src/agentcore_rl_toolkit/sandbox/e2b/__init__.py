"""E2B-compatible sandboxes on AgentCore Runtime.

``AgentCoreE2BProxy`` serves the E2B API on a loopback port, so the unmodified
``e2b`` SDK runs sandboxes as AgentCore Runtime sessions.
"""

from .proxy import AgentCoreE2BProxy

__all__ = ["AgentCoreE2BProxy"]
