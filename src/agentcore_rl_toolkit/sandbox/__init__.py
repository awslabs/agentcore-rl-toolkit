"""Sandbox SDK: run shell commands in arbitrary Docker images on AgentCore Runtime.

Pairs with the ``agentcore-sandboxd`` health shim (see ``sandboxd/`` at the repo
root), which makes any image satisfy the AgentCore Runtime container contract.
Command execution uses AgentCore Runtime's native ``InvokeAgentRuntimeCommand``.
"""

from .client import Sandbox, SandboxClient
from .types import ExecResult, SandboxProtocolError

__all__ = [
    "SandboxClient",
    "Sandbox",
    "ExecResult",
    "SandboxProtocolError",
]
