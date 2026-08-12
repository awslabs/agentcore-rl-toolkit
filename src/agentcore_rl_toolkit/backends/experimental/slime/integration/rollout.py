"""Per-sample rollout function for slime's ``--custom-generate-function-path`` hook.

    --custom-generate-function-path \
        agentcore_rl_toolkit.backends.experimental.slime.integration.rollout.generate
    --custom-reward-post-process-path \
        agentcore_rl_toolkit.backends.experimental.slime.integration.rewards.normalize_episode_rewards
    --custom-config-path /path/to/agentcore.yaml

    Config YAML keys:
        agent_runtime_arn: "arn:aws:bedrock-agentcore:..."
        s3_bucket: "my-bucket"
        gateway_port: 0              # 0 = auto-assign
        max_rollout_time: 1800
        tps_limit: 25
        max_pool_connections: 100
        model_id: null               # defaults to args.hf_checkpoint
        exp_id: null                 # defaults to wandb run identity
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import uuid
from argparse import Namespace
from typing import Any

from slime.utils.http_utils import get_host_info
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from agentcore_rl_toolkit.client import RolloutClient
from agentcore_rl_toolkit.rollout_gateway import BaseTrace, HfTemplateRenderer, ThreadedGatewayServer, TraceRecord
from agentcore_rl_toolkit.rollout_gateway.gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sglang_http import SglangHttpBackend

logger = logging.getLogger(__name__)

_STATUS = {s.value: Sample.Status(s.value) for s in Sample.Status}

_REQUIRED = ("agent_runtime_arn", "s3_bucket")


@dataclasses.dataclass(frozen=True)
class AgentCoreRLConfig:
    agent_runtime_arn: str | None = None
    s3_bucket: str | None = None
    gateway_host: str | None = None
    exp_id: str | None = None
    model_id: str | None = None
    max_rollout_time: float = 1800.0
    tps_limit: int = 25
    max_pool_connections: int = 100
    gateway_port: int = 0

    @classmethod
    def from_args(cls, args: Namespace) -> AgentCoreRLConfig:
        config = cls(
            **{
                f.name: getattr(args, f.name)
                for f in dataclasses.fields(cls)
                if getattr(args, f.name, None) is not None
            }
        )
        missing = [name for name in _REQUIRED if getattr(config, name) is None]
        if missing:
            raise ValueError(
                f"Missing required config key(s): {missing}. " "Set them in the YAML passed to --custom-config-path."
            )
        return config


class AgentCoreRLService(metaclass=SingletonMeta):
    def __init__(self, args: Namespace) -> None:
        self.config = AgentCoreRLConfig.from_args(args)
        self.model_id = self.config.model_id or args.hf_checkpoint
        self.exp_id = self.config.exp_id or str(args.wandb_group or args.wandb_project or "slime-agentcore")

        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        self.gateway = RolloutGateway(
            backend=SglangHttpBackend(router_url),
            renderer=HfTemplateRenderer(tokenizer),
            tokenizer=tokenizer,
        )
        self.server = ThreadedGatewayServer(
            self.gateway,
            host=self.config.gateway_host or get_host_info()[1],  # must be routable from ACR VPC
            port=self.config.gateway_port,
        )
        self.server.start()
        self.base_url = self.server.base_url

        self.client = RolloutClient(
            agent_runtime_arn=self.config.agent_runtime_arn,
            s3_bucket=self.config.s3_bucket,
            exp_id=self.exp_id,
            tps_limit=self.config.tps_limit,
            max_pool_connections=self.config.max_pool_connections,
        )
        logger.info("[agentcore] gateway=%s sglang=%s model_id=%s", self.base_url, router_url, self.model_id)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> list[Sample]:
    """Run one rollout on an ACR agent, returning one Sample per trajectory leaf."""
    service = AgentCoreRLService(args)
    payload = _build_payload(sample)
    sid = str(uuid.uuid4())
    sample.session_id = sid

    service.gateway.create_session(sid, sampling_defaults=sampling_params)

    result: dict[str, Any] = {}
    error: str | None = None
    try:
        future = await service.client.invoke_async(
            payload,
            session_id=sid,
            input_id=str(sample.index),
            base_url=service.base_url,
            model_id=service.model_id,
            api_key=sid,
        )
        result = await future.result_async(timeout=service.config.max_rollout_time)
    except asyncio.TimeoutError:
        error = f"rollout timed out after {service.config.max_rollout_time}s"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    status_code = result.get("status_code")
    if error is None and status_code is not None and status_code != 200:
        error = f"agent returned status_code={status_code}: {result.get('stop_reason', 'unknown')}"
    if error:
        logger.warning("[agentcore] rollout failed (sid=%s): %s", sid, error)

    try:
        records = await service.gateway.finish_session(sid, base_sample=BaseTrace(index=sample.index))
        records = [r for r in records if r.token_ids and r.response_length]
        if not records:
            return [_aborted(sample, error or "agent produced no LLM turns")]

        reward = _agent_reward(result) if error is None else 0.0
        return [_to_sample(sample, r, reward) for r in records]
    finally:
        await service.gateway.drop_session(sid)


def _build_payload(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata or {}
    payload = metadata.get("payload")
    if isinstance(payload, dict):
        return payload
    raise ValueError(
        "Dataset row has no `payload` field in its metadata. " 'Author rows as {"metadata": {"payload": {...}}}.'
    )


def _agent_reward(result: dict[str, Any]) -> float | None:
    """Returns the agent's reward, or None to let slime's rm_hub score instead."""
    rewards = result.get("rewards")
    if rewards is None or (isinstance(rewards, list) and not rewards):
        return None
    value = rewards[-1] if isinstance(rewards, list) else rewards
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Non-numeric reward from agent: rewards={rewards!r}")
    return float(value)


def _to_sample(base: Sample, record: TraceRecord, reward: float | None) -> Sample:
    return Sample(
        index=base.index,
        group_index=base.group_index,
        rollout_id=base.index,
        prompt=base.prompt,
        label=base.label,
        session_id=base.session_id,
        tokens=list(record.token_ids),
        response=record.response,
        response_length=record.response_length,
        loss_mask=list(record.loss_mask),
        rollout_log_probs=list(record.logprobs),
        reward=reward,
        status=_STATUS[record.status.value],
        metadata={**(base.metadata or {}), "acr_session_id": base.session_id},
    )


def _aborted(base: Sample, reason: str) -> Sample:
    logger.warning("[agentcore] sample %s aborted: %s", base.index, reason)
    base.tokens = [0, 0]
    base.response = ""
    base.response_length = 1
    base.loss_mask = [0]
    base.rollout_log_probs = [0.0]
    base.reward = 0.0
    base.rollout_id = base.index
    base.remove_sample = True
    base.status = Sample.Status.ABORTED
    base.metadata = {**(base.metadata or {}), "abort_reason": reason}
    return base
