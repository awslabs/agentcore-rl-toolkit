"""Fairly distribute successive agent-loop dispatches across Ray workers."""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import wraps
from typing import Any

_NEXT_WORKER_INDEX_ATTR = "_agentcore_next_agent_loop_worker_index"
_PATCHED_ATTR = "_agentcore_round_robin_dispatch"
_DISPATCH_LOCK = threading.Lock()


def _generate_sequences_round_robin(
    manager: Any, prompts: Any, upstream_generate_sequences: Callable[[Any, Any], None]
) -> None:
    """Rotate chunk zero's worker, then delegate all dispatch behavior to verl."""
    workers = manager.agent_loop_workers
    dispatched_worker_count = len(prompts.chunk(len(workers)))
    with _DISPATCH_LOCK:
        start_index = getattr(manager, _NEXT_WORKER_INDEX_ATTR, 0)
        manager.agent_loop_workers = [*workers[start_index:], *workers[:start_index]]
        try:
            upstream_generate_sequences(manager, prompts)
        finally:
            manager.agent_loop_workers = workers
        setattr(manager, _NEXT_WORKER_INDEX_ATTR, (start_index + dispatched_worker_count) % len(workers))


def install_round_robin_agent_loop_dispatch() -> None:
    """Replace the singleton-biased v1 TQ dispatcher once per Python process."""
    from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ

    if getattr(AgentLoopManagerTQ.generate_sequences, _PATCHED_ATTR, False):
        return

    upstream_generate_sequences = AgentLoopManagerTQ.generate_sequences

    @wraps(upstream_generate_sequences)
    def generate_sequences(self: Any, prompts: Any) -> None:
        _generate_sequences_round_robin(self, prompts, upstream_generate_sequences)

    setattr(generate_sequences, _PATCHED_ATTR, True)
    AgentLoopManagerTQ.generate_sequences = generate_sequences


class RoundRobinAgentLoopDispatchMixin:
    """Install fair agent-loop dispatch before the concrete verl trainer initializes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        install_round_robin_agent_loop_dispatch()
        super().__init__(*args, **kwargs)
