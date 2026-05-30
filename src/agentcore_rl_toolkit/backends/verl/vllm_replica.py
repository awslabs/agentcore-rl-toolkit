"""AgentCore-specific vLLM server and replica.

Prevents ghost AgentCore requests from corrupting the vLLM engine during
HYBRID-mode sleep/wake cycles by using vLLM's built-in pause_generation()
/ resume_generation() to gate the HTTP server around sleep transitions.
"""

import logging
from typing import Any, Callable

import ray
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AgentCoreVLLMHttpServer(vLLMHttpServer):
    """vLLM HTTP server with ghost-request protection for AgentCore rollout."""

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        result = await super().abort_all_requests(reset_prefix_cache=reset_prefix_cache)

        # Always issue a prefix-cache reset as a synchronous barrier: the
        # engine core processes ZMQ messages in order, so by the time this
        # returns all prior ABORTs are guaranteed to have been processed.
        if reset_prefix_cache and result.get("aborted_count", 0) == 0:
            await self.clear_kv_cache()

        return result

    async def sleep(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            await self.engine.pause_generation()
        await super().sleep()

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        await super().collective_rpc(method, timeout, args, kwargs)

        if method == "wake_up" and self.rollout_mode == RolloutMode.HYBRID:
            tags = (kwargs or {}).get("tags", None)
            if tags is None or "kv_cache" in tags:
                await self.engine.resume_generation()
                logger.info("Resumed generation after wake_up (tags=%s)", tags)


class AgentCoreVLLMReplica(vLLMReplica):
    """vLLM replica that uses AgentCoreVLLMHttpServer for ghost-request protection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(AgentCoreVLLMHttpServer)
