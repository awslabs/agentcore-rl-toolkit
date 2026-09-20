"""Sandbox SDK: run shell commands in arbitrary Docker images on AgentCore Runtime.

Pairs with ``agentcore-sandboxd`` (see ``sandboxd/`` at the repo root), which
owns foreground/background commands and their persisted results.
"""

from .client import ExecError, ExecHandle, Sandbox, SandboxClient
from .types import ExecResult, SandboxProtocolError

__all__ = [
    "SandboxClient",
    "Sandbox",
    "ExecResult",
    "ExecHandle",
    "ExecError",
    "SandboxProtocolError",
]
