"""A rollout session that drives the agent over A2A on a Bedrock AgentCore runtime.

Speaks JSON-RPC to the agent's A2A server over the runtime's ``…/invocations/`` URL, SigV4-signed.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

from a2a.client import Client

from agentcore_rl_toolkit.aws_tools.agentcore_tools import (
    region_of,
    shared_agentcore_client,
    start_agentcore_session,
    stop_agentcore_instance_session,
)
from agentcore_rl_toolkit.aws_tools.boto3_tools import shared_sigv4_httpx_client
from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict, measure_span_persistent
from agentcore_rl_toolkit.rollout_session.a2a_client import A2ARolloutSession, build_a2a_client

logger = logging.getLogger(__name__)

# bedrock-agentcore data plane; the same service the boto3 client signs against.
ACR_SERVICE = "bedrock-agentcore"


def runtime_invocations_url(runtime_arn: str) -> str:
    """The A2A JSON-RPC URL for ``runtime_arn`` (trailing slash: the A2A rpc endpoint)."""
    region = region_of(runtime_arn)
    encoded = quote(runtime_arn, safe="")
    return f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{encoded}/invocations/"


class AgentCoreA2ASession(A2ARolloutSession):
    """A2A rollout session backed by an ACR runtime session."""

    def __init__(
        self,
        session_id: str,
        session_state: PersistentDict,
        runtime_arn: str,
        capacity_provider_arn: str,
    ):
        super().__init__(session_id, session_state)
        # Runtime and provider must share a region (they always do); checked here because
        # nothing later reads the provider region.
        self.region = region_of(runtime_arn)
        assert region_of(capacity_provider_arn) == self.region, "runtime and capacity provider are in two regions"
        self.runtime_arn = runtime_arn
        self.capacity_provider_arn = capacity_provider_arn
        self._url = runtime_invocations_url(runtime_arn)

    async def _client(self) -> Client:
        return build_a2a_client(await shared_sigv4_httpx_client(self.region, ACR_SERVICE), self._url)

    async def _acr(self) -> Any:
        return await shared_agentcore_client(self.region)

    async def setup(self, task: dict) -> None:
        await self.session_state.update(
            {
                "capacity_provider_arn": self.capacity_provider_arn,
                "runtime_arn": self.runtime_arn,
            }
        )
        async with measure_span_persistent("agentcore_setup", self.session_state):
            # Warm the microVM via the boto3 data plane: it has the readiness-race retry the A2A path lacks.
            await start_agentcore_session(self.runtime_arn, self.session_id, await self._acr())

        async with measure_span_persistent("task_setup", self.session_state):
            await super().setup(task)

    async def shutdown(self) -> None:
        # Let the agent clean up a mid-flight rollout before the session is stopped.
        try:
            await self._cancel_if_unfinished()
        except Exception as e:  # don't let cancel mask the teardown or the real error
            logger.warning("cancel-on-shutdown for %s failed: %s: %s", self.session_id, type(e).__name__, e)
        await stop_agentcore_instance_session(self.capacity_provider_arn, self.session_id, await self._acr())
