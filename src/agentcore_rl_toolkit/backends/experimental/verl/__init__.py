"""Experimental verl integration built on the in-repo rollout gateway.

Plugs into verl as a custom agent loop (``rollout.agent.agent_loop_config_path``)
running under the stock ``python -m verl.trainer.main_ppo`` entrypoint with
``trainer.use_v1=true``. See README.md in this directory for setup.

``AgentCoreAgentLoop`` (which imports verl) is exposed lazily so that importing
this package — e.g. for ``VerlSamplingBackend`` in tests — does not require verl.
"""

from .sampling_backend import VerlSamplingBackend


def __getattr__(name: str):
    if name == "AgentCoreAgentLoop":
        from .agent_loop import AgentCoreAgentLoop

        return AgentCoreAgentLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgentCoreAgentLoop", "VerlSamplingBackend"]
