"""Rollouts over the Runtime HTTP contract, using one shared client."""

import logging
import uuid

from agentcore_rl_toolkit.aws_tools.agentcore_tools import region_of, shared_agentcore_client
from agentcore_rl_toolkit.aws_tools.persistent_dict import PersistentDict
from agentcore_rl_toolkit.runtime import AgentCoreHttpClient, InvocationError

from .errors import RolloutContractError
from .exception_utils import exception_to_string
from .lifecycle import RolloutSession
from .wire import RolloutDumpResponse

logger = logging.getLogger(__name__)


class AgentCoreHttpSession(RolloutSession):
    """One Runtime session. The caller owns the application payload.

    Constructed from config, a session invokes through the process-wide
    ``bedrock-agentcore`` client (:func:`shared_agentcore_client`). A caller that owns
    its client uses :meth:`with_client` instead.
    """

    def __init__(
        self,
        session_id: str,
        session_state: PersistentDict,
        *,
        agentcore_runtime_arn: str,
    ):
        region = region_of(agentcore_runtime_arn)
        client = AgentCoreHttpClient(agentcore_runtime_arn, aws_client=lambda: shared_agentcore_client(region))
        self._init(session_id, session_state, client)

    @classmethod
    def with_client(
        cls,
        session_id: str,
        client: AgentCoreHttpClient,
        session_state: PersistentDict | None = None,
    ) -> "AgentCoreHttpSession":
        """A session over a caller-owned client, which the caller also closes."""
        session = cls.__new__(cls)
        session._init(session_id, session_state, client)
        return session

    def _init(self, session_id: str, session_state: PersistentDict | None, client: AgentCoreHttpClient) -> None:
        self.session_id = session_id
        self._client = client
        self._session_state = session_state
        self._started = False

    async def __aenter__(self) -> "AgentCoreHttpSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    async def setup(self, task: dict) -> None:
        """No-op for this adapter."""

    async def run(self, task: dict) -> RolloutDumpResponse:
        invocation_id = uuid.uuid4().hex
        if self._session_state is not None:
            await self._session_state.update({"runtime_arn": self._client.runtime_arn, "invocation_id": invocation_id})
        # A failed submission can still have created a session.
        self._started = True
        try:
            handle = await self._client.invoke(
                task, session_id=self.session_id, invocation_id=invocation_id, background=True
            )
            result = await handle.result()
        except InvocationError as exc:
            return RolloutDumpResponse(task_output=None, reward=None, exception=exception_to_string(exc))
        return to_dump(result)

    async def shutdown(self) -> None:
        """Stop this session once; cleanup failures must not hide rollout failures."""
        if self._started:
            self._started = False
            try:
                await self._client.stop_session(self.session_id)
            except Exception:
                logger.warning("Failed to stop Runtime session %s", self.session_id, exc_info=True)


def to_dump(result: object) -> RolloutDumpResponse:
    if not isinstance(result, dict):
        raise RolloutContractError("An HTTP rollout must return a JSON object")
    return RolloutDumpResponse(
        task_output=result, reward=result.get("reward"), metrics=result.get("metrics", {}), exception=None
    )
