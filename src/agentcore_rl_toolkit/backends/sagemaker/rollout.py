"""ACR rollout utilities shared across training algorithms."""

import asyncio
import logging
import socket
import uuid
from typing import Any

from agentcore_rl_toolkit.rollout_gateway import BaseTrace, TraceRecord
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession
from agentcore_rl_toolkit.runtime import AgentCoreHttpClient

logger = logging.getLogger(__name__)


def local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


def load_dataset(path: str) -> list[dict]:
    """Load ACR payloads from a parquet or JSONL file.

    Parquet: expects a ``payload`` column.
    JSONL: expects each row to have a ``payload`` field.
    """
    if path.endswith(".parquet"):
        import pandas as pd

        return pd.read_parquet(path)["payload"].tolist()
    import json

    payloads = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                payloads.append(row["payload"])
    return payloads


async def run_one_rollout(
    *,
    client: AgentCoreHttpClient,
    gateway,
    payload: dict,
    base_url: str,
    model_id: str,
    max_rollout_time: float,
    sampling_defaults: dict,
    max_context_tokens: int,
) -> tuple[list[TraceRecord], float, dict]:
    """Run one ACR rollout against an ``AgentCoreRuntimeApp`` agent; return (records, reward, result).

    Creates a gateway session keyed by a fresh UUID (= ACR runtimeSessionId =
    api_key for trajectory capture), invokes the agent with ``_config`` (base_url,
    model_id, api_key), awaits its result over HTTP, and drains the session into
    TraceRecords. The agent's scalar ``reward`` is passed to finish_session so
    TraceRecord.reward is set correctly, and returned to the caller for advantage
    computation. Returns reward=0.0 on failure.
    """
    sid = str(uuid.uuid4())
    gateway.create_session(
        sid,
        sampling_defaults=sampling_defaults,
        max_context_tokens=max_context_tokens,
    )

    result: dict[str, Any] = {}
    task = {**payload, "_config": {"base_url": base_url, "model_id": model_id, "api_key": sid}}
    try:
        async with AgentCoreHttpSession.with_client(sid, client) as session:
            dump = await asyncio.wait_for(session.run(task), max_rollout_time)
        if dump.exception is not None:
            raise RuntimeError(dump.exception)
        result = dump.task_output
    except asyncio.TimeoutError:
        logger.warning("ACR rollout timed out after %ss (sid=%s)", max_rollout_time, sid)
    except Exception as e:
        logger.warning("ACR rollout failed (sid=%s): %s: %s", sid, type(e).__name__, e)

    reward = _extract_reward(result)
    records = await gateway.finish_session(sid, base_sample=BaseTrace(rollout_id=sid), reward=reward)
    return [r for r in records if r.token_ids], reward, result


def _extract_reward(result: dict) -> float:
    try:
        return float(result["reward"])
    except (KeyError, TypeError, ValueError):
        return 0.0
