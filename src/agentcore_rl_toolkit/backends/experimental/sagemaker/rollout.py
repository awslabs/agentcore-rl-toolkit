"""ACR rollout utilities shared across training algorithms."""

import asyncio
import logging
import socket
import uuid
from typing import Any

from agentcore_rl_toolkit.client import RolloutClient
from agentcore_rl_toolkit.rollout_gateway import BaseTrace, TraceRecord

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
    client: RolloutClient,
    gateway,
    payload: dict,
    base_url: str,
    model_id: str,
    max_rollout_time: float,
    sampling_defaults: dict,
    max_context_tokens: int,
) -> tuple[list[TraceRecord], float, dict]:
    """Run one ACR rollout; return (records, reward, s3_result).

    Creates a gateway session keyed by a fresh UUID (= ACR runtimeSessionId =
    api_key for trajectory capture), invokes the ACR agent, awaits the S3 result,
    and drains the session into TraceRecords. The reward from the S3 result is
    passed to finish_session so TraceRecord.reward is set correctly, and returned
    to the caller for advantage computation. Returns reward=0.0 on failure.
    """
    sid = str(uuid.uuid4())
    gateway.create_session(
        sid,
        sampling_defaults=sampling_defaults,
        max_context_tokens=max_context_tokens,
    )

    s3_result: dict[str, Any] = {}
    try:
        future = await client.invoke_async(
            payload,
            session_id=sid,
            input_id=sid,
            base_url=base_url,
            model_id=model_id,
            api_key=sid,
        )
        s3_result = await future.result_async(timeout=max_rollout_time)
    except asyncio.TimeoutError:
        logger.warning("ACR rollout timed out after %ss (sid=%s)", max_rollout_time, sid)
    except Exception as e:
        logger.warning("ACR rollout failed (sid=%s): %s: %s", sid, type(e).__name__, e)

    status = s3_result.get("status_code")
    if status is not None and status != 200:
        logger.warning("ACR rollout failed (sid=%s): agent status_code=%s", sid, status)

    reward = _extract_reward(s3_result)
    records = await gateway.finish_session(sid, base_sample=BaseTrace(rollout_id=sid), reward=reward)
    return [r for r in records if r.token_ids], reward, s3_result


def _extract_reward(s3_result: dict) -> float:
    rewards = s3_result.get("rewards")
    if rewards is None:
        return 0.0
    if isinstance(rewards, list):
        return float(rewards[-1]) if rewards else 0.0
    try:
        return float(rewards)
    except (TypeError, ValueError):
        return 0.0
