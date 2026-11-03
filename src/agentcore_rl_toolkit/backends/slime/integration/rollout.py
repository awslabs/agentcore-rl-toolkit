"""Per-sample rollout function for slime's ``--custom-generate-function-path`` hook.

    --custom-generate-function-path \
        agentcore_rl_toolkit.backends.slime.integration.rollout.generate
    --custom-reward-post-process-path \
        agentcore_rl_toolkit.backends.slime.integration.rewards.normalize_episode_rewards
    --custom-config-path /path/to/agentcore.yaml

    Config YAML keys:
        rollout_session_backend:     # passed to rollout_session.factory.make_session
          backend: agentcore_rl_toolkit.rollout_session.agentcore_http_session.AgentCoreHttpSession
          agentcore_runtime_arn: "arn:aws:bedrock-agentcore:..."
        gateway_port: 0              # 0 = auto-assign
        max_rollout_time: 1800
        tps_limit: 25                # new rollout sessions per second
        model_id: null               # defaults to args.hf_checkpoint

Each rollout runs ``{**metadata.payload, "_config": {base_url, model_id, api_key}}``
in its own ``RolloutSession`` and reads a scalar ``reward`` from the returned dict.
With ``AgentCoreHttpSession``, the agent is an ``AgentCoreRuntimeApp``.

With slime's ``--use-score-centering`` (``--rollout-top-p 1``), the gateway captures the
sampler's top-``--score-centering-top-k`` head for every trained token, and each turn's
sampling params must pass slime's ``score_centering_request`` checks.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import uuid
from argparse import Namespace
from typing import Any

from pydantic import ValidationError
from slime.utils.http_utils import get_host_info
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister, PersistentDict
from agentcore_rl_toolkit.concurrency.rate_limiter import LocalRateLimiter
from agentcore_rl_toolkit.rollout_gateway import BaseTrace, HfTemplateRenderer, ThreadedGatewayServer, TraceRecord
from agentcore_rl_toolkit.rollout_gateway.gateway import RolloutGateway
from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sglang_http import SglangHttpBackend
from agentcore_rl_toolkit.rollout_session.errors import RolloutContractError
from agentcore_rl_toolkit.rollout_session.factory import make_session, resolve_backend

logger = logging.getLogger(__name__)

_STATUS = {s.value: Sample.Status(s.value) for s in Sample.Status}

_REQUIRED = ("rollout_session_backend",)


@dataclasses.dataclass(frozen=True)
class AgentCoreRLConfig:
    rollout_session_backend: dict[str, Any] | None = None
    gateway_host: str | None = None
    model_id: str | None = None
    max_rollout_time: float = 1800.0
    tps_limit: int = 25
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

        # Fail at startup, not on every rollout, if the backend path does not resolve.
        resolve_backend(self.config.rollout_session_backend["backend"])
        self.session_rate_limiter = LocalRateLimiter(self.config.tps_limit)
        logger.info("[agentcore] gateway=%s sglang=%s model_id=%s", self.base_url, router_url, self.model_id)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> list[Sample]:
    """Run one rollout on an ACR agent, returning one Sample per trajectory leaf."""
    service = AgentCoreRLService(args)
    # slime clears use_score_centering in the args it passes for evaluation.
    top_k = _score_centering_top_k(args)
    payload = _build_payload(sample)
    sid = str(uuid.uuid4())
    sample.session_id = sid
    # The session id doubles as the gateway's trajectory key, carried in the api-key slot.
    task = {**payload, "_config": {"base_url": service.base_url, "model_id": service.model_id, "api_key": sid}}

    service.gateway.create_session(
        sid,
        sampling_defaults=sampling_params,
        num_top_logprobs=top_k,
        validate_sampling_params=_score_centering_validator(args) if top_k else None,
    )
    try:
        result, error = await _run_agent(service, sid, task)
        if error:
            logger.warning("[agentcore] rollout failed (sid=%s): %s", sid, error)

        records = await service.gateway.finish_session(sid, base_sample=BaseTrace(index=sample.index))
        records = [r for r in records if r.token_ids and r.response_length]
        if not records:
            return [_aborted(sample, error or "agent produced no LLM turns", args, top_k)]

        reward = _agent_reward(result) if error is None else 0.0
        return [_to_sample(sample, r, reward, args, top_k) for r in records]
    finally:
        await service.gateway.drop_session(sid)


async def _run_agent(service: AgentCoreRLService, sid: str, task: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Run the agent in its own rollout session; return (result, error).

    Leaving the session context shuts the session down, including on timeout.
    A bad backend config or malformed agent response raises, since it would recur
    on every rollout.
    """
    state = PersistentDict({"session_id": sid}, persister=NullPersister())
    session = make_session(sid, service.config.rollout_session_backend, state)
    try:
        async with session:
            await service.session_rate_limiter.wait_async()
            async with asyncio.timeout(service.config.max_rollout_time):
                await session.setup(task)
                dump = await session.run(task)
    except TimeoutError:
        return {}, f"rollout timed out after {service.config.max_rollout_time}s"
    except (RolloutContractError, ValidationError):
        raise
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"
    return dump.task_output or {}, dump.exception


def _build_payload(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata or {}
    payload = metadata.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(
            "Dataset row has no `payload` field in its metadata. " 'Author rows as {"metadata": {"payload": {...}}}.'
        )
    if "_config" in payload:
        raise ValueError("Dataset payload must not contain reserved field '_config'")
    return payload


def _agent_reward(result: dict[str, Any]) -> float | None:
    """Returns the agent's reward, or None to let slime's rm_hub score instead."""
    reward = result.get("reward")
    if reward is None:
        return None
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError(f"Non-numeric reward from agent: reward={reward!r}")
    return float(reward)


def _score_centering_top_k(args: Namespace) -> int:
    """The sampler head width slime's score centering trains on, or 0 when it is off."""
    if not getattr(args, "use_score_centering", False):
        return 0
    if args.rollout_top_p < 1:
        raise ValueError("Exact top-p score centering is not supported through the gateway; use --rollout-top-p 1.")
    return args.score_centering_top_k


def _score_centering_validator(args: Namespace):
    from slime.utils.score_centering import score_centering_request

    return functools.partial(score_centering_request, args)


def _to_sample(
    base: Sample, record: TraceRecord, reward: float | None, args: Namespace | None = None, top_k: int = 0
) -> Sample:
    sample = Sample(
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
    if top_k:
        _append_sampler_heads(sample, record, args, top_k)
    return sample


def _append_sampler_heads(sample: Sample, record: TraceRecord, args: Namespace, top_k: int) -> None:
    """Rebuild the response through slime's ``append_response_tokens``, one segment at a time.

    Trained segments carry their sampler head as ``score_centering_topk``; slime fills
    placeholder rows for the masked segments between them.
    """
    prompt_length = len(record.token_ids) - record.response_length
    response = record.token_ids[prompt_length:]
    sample.tokens = list(record.token_ids[:prompt_length])
    sample.response_length = 0
    sample.loss_mask = None
    sample.rollout_log_probs = None
    end = 0
    for offset, head in record.top_logprobs:
        if offset > end:
            sample.append_response_tokens(
                args, tokens=response[end:offset], trainable=False, update_terminal_info=False
            )
        end = offset + len(head)
        if head.ids.shape != (end - offset, top_k):
            raise RuntimeError(f"sampler head has shape {head.ids.shape}, expected {(end - offset, top_k)}")
        sample.append_response_tokens(
            args,
            tokens=response[offset:end],
            log_probs=record.logprobs[offset:end],
            meta_info={"score_centering_topk": (head.ids, head.logprobs)},
            update_terminal_info=False,
        )
    if end < len(response):
        sample.append_response_tokens(args, tokens=response[end:], trainable=False, update_terminal_info=False)
    if sample.loss_mask != list(record.loss_mask):
        raise RuntimeError("trained tokens without a sampler head")


def _aborted(base: Sample, reason: str, args: Namespace | None = None, top_k: int = 0) -> Sample:
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
    if top_k:
        # slime batches the sampler heads of every sample, so this one needs placeholders.
        base.tokens, base.response_length, base.loss_mask, base.rollout_log_probs = [0], 0, None, None
        base.rollout_topk_token_ids = base.rollout_topk_log_probs = None
        base.append_response_tokens(args, tokens=[0], trainable=False, update_terminal_info=False)
    return base
