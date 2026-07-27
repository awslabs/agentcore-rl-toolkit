"""``AgentCoreAgentLoop`` — verl agent loop that runs rollouts on Bedrock
AgentCore Runtime with token capture through the in-repo rollout gateway.

Per rollout: create a gateway session keyed by a fresh uuid → invoke the ACR
agent with that uuid as the ACR ``runtimeSessionId`` (the container places it in
its LLM client's api-key slot, which the gateway reads as the Bearer sid) → await
the agent's S3 result (the completion signal; may carry an inline reward) →
drain the gateway session into TraceRecords → reshape each into an
``AgentLoopOutput``.

Registered as ``agentcore_agent`` — enable via verl's
``rollout.agent.agent_loop_config_path`` under the stock
``python -m verl.trainer.main_ppo`` entrypoint. Requires ``trainer.use_v1=true``:
``run`` returns ``list[AgentLoopOutput]`` (one per trajectory-tree leaf, e.g.
sub-agent forks), which only the v1 TransferQueue path consumes.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from agentcore_rl_toolkit.client import RolloutClient
from agentcore_rl_toolkit.rollout_gateway import BaseTrace, TraceRecord

from .gateway_host import GatewayHandle, get_or_start_gateway

logger = logging.getLogger(__name__)

# Agent loops are instantiated per trajectory, so same-config instances MUST share
# one client: the client owns the ACRRateLimiter, and a fresh limiter per instance
# means no effective rate limiting toward ACR's per-ARN TPS cap. Keyed by config
# rather than a process singleton so distinct ARNs get distinct clients. Only ever
# touched from the AgentLoopWorker's asyncio thread (RolloutClient isn't thread-safe).
_CLIENTS: dict[tuple, RolloutClient] = {}


def _get_or_create_client(
    *, agent_runtime_arn: str, s3_bucket: str, exp_id: str, tps_limit: int, max_pool_connections: int
) -> RolloutClient:
    key = (agent_runtime_arn, s3_bucket, exp_id, tps_limit, max_pool_connections)
    client = _CLIENTS.get(key)
    if client is None:
        client = _CLIENTS[key] = RolloutClient(
            agent_runtime_arn=agent_runtime_arn,
            s3_bucket=s3_bucket,
            exp_id=exp_id,
            tps_limit=tps_limit,
            max_pool_connections=max_pool_connections,
        )
    return client


def _reset_client_for_tests() -> None:
    _CLIENTS.clear()


def _extract_agent_reward(result: dict) -> float | None:
    """The agent-reported reward from a session result (the ``{"rewards": ...}``
    convention of ``@rollout_entrypoint`` apps: scalar, or last element of a
    list), or ``None`` if the agent didn't report one.

    A non-numeric value raises rather than scoring 0.0: broken reward code is broken
    on every rollout, and zeros would flatten every GRPO group's advantages instead.
    verl contains the raise per prompt group, so training continues.
    """
    rewards = result.get("rewards")
    if rewards is None:
        return None
    if isinstance(rewards, list) and not rewards:
        return None  # an empty list reports no reward
    value = rewards[-1] if isinstance(rewards, list) else rewards
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"The agent returned a non-numeric built-in reward: rewards={rewards!r} ({e}). "
            "It must be a float, or a list of floats whose last element is the reward — "
            "see the reward contract in backends/experimental/verl/README.md."
        ) from e


# Deliberately NOT decorated with verl's @register: it stores a bare
# {"_target_": ...} with no kwargs, and fires when hydra imports this module —
# overwriting the YAML entry that carries our required kwargs. The first rollout
# would work, every later one crash on missing kwargs. Registration is YAML-only.
class AgentCoreAgentLoop(AgentLoopBase):
    """Runs each rollout on an ACR-deployed agent, capturing token-level
    trajectories through the process-local rollout gateway."""

    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        data_config,
        *,
        agent_runtime_arn: str,
        s3_bucket: str,
        exp_id: str | None = None,
        tps_limit: int = 5,
        max_pool_connections: int = 100,
        max_rollout_time: float = 1800.0,
        gateway_bind_host: str = "0.0.0.0",
        gateway_port: int = 0,
        gateway_public_host: str | None = None,
        gateway_adapters: list[str] | None = None,
        max_turns_per_sid: int | None = None,
        fork_threshold_tokens: int | None = None,
        reward_mode: str = "built_in",
        **kwargs,  # swallows the YAML entry's `name`, verl's `tools`, and future kwargs
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config, **kwargs)

        if not self.config.trainer.get("use_v1", False):
            raise ValueError(
                "AgentCoreAgentLoop requires trainer.use_v1=true: it returns "
                "list[AgentLoopOutput] (one per trajectory-tree leaf), which only "
                "the v1 TransferQueue path consumes."
            )
        if reward_mode == "separate":
            # The reward managers index those columns *before* merging acr_result into
            # extra_info, so a payload-only row KeyErrors before the reward fn runs.
            raise ValueError(
                "reward_mode='separate' is not supported yet: verl's reward managers require "
                "`data_source` and `reward_model.ground_truth` dataset columns, which the "
                "payload-first dataset contract does not provide. Use reward_mode='built_in' "
                "and have the agent return {'rewards': ...} in its session result."
            )
        if reward_mode != "built_in":
            raise ValueError(f"reward_mode must be 'built_in', got {reward_mode!r}")

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_rollout_time = max_rollout_time
        # Kept as config, not inlined, so a validated trainer-side mode can land
        # without a signature change. Reward semantics: see the README.
        self.reward_mode = reward_mode
        self.model_id = self.config.actor_rollout_ref.model.path

        self._gateway: GatewayHandle = get_or_start_gateway(
            server_manager=server_manager,
            tokenizer=tokenizer,
            host=gateway_bind_host,
            port=gateway_port,
            public_host=gateway_public_host,
            adapters=gateway_adapters,
            max_turns_per_sid=max_turns_per_sid,
            fork_threshold_tokens=fork_threshold_tokens,
        )
        # Default exp_id must be identical across all AgentLoopWorker processes of
        # one run, so derive it from verl's run identity instead of inventing one.
        trainer_cfg = self.config.trainer
        project = trainer_cfg.get("project_name", "verl")
        experiment = trainer_cfg.get("experiment_name", "run")
        self._exp_id = exp_id or f"{project}-{experiment}"
        self._client = _get_or_create_client(
            agent_runtime_arn=agent_runtime_arn,
            s3_bucket=s3_bucket,
            exp_id=self._exp_id,
            tps_limit=tps_limit,
            max_pool_connections=max_pool_connections,
        )

    # Returning a list deliberately widens AgentLoopBase.run's annotation: the v1
    # TQ path accepts AgentLoopOutput | list[AgentLoopOutput] (one row per
    # trajectory-tree leaf); __init__ asserts trainer.use_v1 accordingly.
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:  # type: ignore[override]
        sid = str(uuid.uuid4())  # gateway Bearer sid == ACR runtimeSessionId (36 chars >= ACR's 33 min)
        start = time.monotonic()

        # Built before any session/ACR state exists: a payload-contract violation
        # is a config error that would hit every rollout — raise it loudly rather
        # than degrading each rollout into an inert row.
        payload = self._build_payload(kwargs)

        self._gateway.gateway.create_session(
            sid,
            sampling_defaults=self._sampling_defaults(sampling_params),
            max_context_tokens=self.prompt_length + self.response_length,
        )

        result: dict[str, Any] = {}
        error: str | None = None
        try:
            future = await self._client.invoke_async(
                payload,
                session_id=sid,
                input_id=str(kwargs.get("uid", sid)),
                # OpenAI-SDK convention: base_url includes the /v1 prefix (the
                # client appends /chat/completions); agents pass it verbatim.
                # TODO: not directly usable by Anthropic-SDK agents (that SDK
                # appends /v1/messages without normalizing an existing /v1).
                base_url=f"{self._gateway.base_url}/v1",
                model_id=self.model_id,
            )
            result = await future.result_async(timeout=self.max_rollout_time)
        except asyncio.TimeoutError:
            error = f"rollout timed out after {self.max_rollout_time}s"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        if error:
            logger.warning("ACR rollout failed (sid=%s): %s", sid, error)

        status_code = result.get("status_code")
        if status_code is not None and status_code != 200 and error is None:
            # Agent-side failure saved to S3; a partial trace may still exist.
            error = f"agent returned status_code={status_code}: {result.get('stop_reason', 'unknown')}"
            logger.warning("ACR rollout failed (sid=%s): %s", sid, error)

        num_turns = self._gateway.gateway.manager.turn_count(sid)
        records = await self._gateway.gateway.finish_session(sid, base_sample=BaseTrace(rollout_id=sid), reward=0.0)
        records = [r for r in records if r.token_ids]
        engine_extra = self._gateway.backend.pop_extra_fields(sid)
        elapsed = time.monotonic() - start

        if not records:
            self._warn_if_static_session_capture(sid)
            return [self._degenerate_output(kwargs, sid, error or "agent produced no LLM turns", elapsed, result)]

        reward = self._resolve_reward(result, error, sid)

        # v1's staleness metrics do int(tag["min_global_steps"]) — the tags must
        # always be real ints. The engine's extra_fields carry them for turns it
        # served; default to the dataloader step otherwise.
        global_steps = int(kwargs.get("global_steps", 0))
        shared_extra = {
            # the session result is parsed JSON (RolloutFuture json.loads it) —
            # already plain python, no sanitizing needed
            "acr_result": result,
            "acr_session_id": sid,
            "num_trace_records": len(records),
            **({"acr_error": error} if error else {}),
            "min_global_steps": global_steps,
            "max_global_steps": global_steps,
            **{k: v for k, v in engine_extra.items() if v is not None},
        }
        if len(records) > 1:
            logger.info(
                "session %s forked into %d trace records (trained tokens per record: %s)",
                sid,
                len(records),
                [sum(r.loss_mask) for r in records],
            )
            # verl scores/broadcasts from outputs[-1]; put the primary
            # (most-trained) record last, keeping tree order otherwise.
            primary = max(range(len(records)), key=lambda i: sum(records[i].loss_mask))
            records.append(records.pop(primary))

        outputs = [
            self._record_to_output(r, i, reward, num_turns, shared_extra, elapsed) for i, r in enumerate(records)
        ]
        return outputs

    # -- helpers ---------------------------------------------------------------

    def _resolve_reward(self, result: dict[str, Any], error: str | None, sid: str) -> float:
        """The rollout's reward_score, which becomes rm_scores directly (verl
        skips reward computation for this rollout). The agent owns scoring;
        failures and contract violations score 0."""
        if error is not None:
            return 0.0
        agent_reward = _extract_agent_reward(result)
        if agent_reward is None:
            logger.warning(
                "The agent returned no {'rewards': ...} for rollout %s; scoring 0.0. "
                "The agent owns scoring — return the reward in its session result "
                "(see the reward contract in backends/experimental/verl/README.md).",
                sid,
            )
            return 0.0
        return agent_reward

    def _warn_if_static_session_capture(self, sid: str) -> None:
        """Diagnose the stale-agent-image failure mode (see the warning below).

        Warn only, never drop the session: the adapters accept unseen keys by design
        (that is how local runs work) and "EMPTY" is legitimate for local/eval traffic.
        Without this, the misconfiguration trains nothing, silently."""
        manager = self._gateway.gateway.manager
        for static_sid in ("EMPTY", "default"):
            if manager.turn_count(static_sid):
                logger.warning(
                    "Rollout %s captured no trace, but turns are accumulating under the "
                    "static session %r — the deployed agent is likely sending a fixed "
                    "api_key instead of context.session_id (stale agent image?). "
                    "See the agent-side contract in backends/experimental/verl/README.md.",
                    sid,
                    static_sid,
                )
                break

    def _sampling_defaults(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        defaults: dict[str, Any] = {"max_new_tokens": self.response_length}
        for key in ("temperature", "top_p", "top_k"):
            if key in sampling_params:
                defaults[key] = sampling_params[key]
        return defaults

    def _build_payload(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """The ACR invoke payload: the row's ``payload`` column, forwarded verbatim.

        The single contract, with no field-selection or forward-everything fallback —
        the row namespace is shared with verl's plumbing fields, and verl-shaped
        columns (chat-format prompts) are not agent-shaped. See the README.
        """
        payload = kwargs.get("payload")
        if isinstance(payload, dict):
            return payload
        raise ValueError(
            "Cannot build the agent payload: the dataset row has no `payload` column. "
            "Author rows with a `payload` column holding the agent's exact invoke "
            "payload (see PayloadDataset and the backend README, which includes a "
            "snippet for converting existing datasets)."
        )

    def _record_to_output(
        self,
        record: TraceRecord,
        index: int,
        reward: float | None,
        num_turns: int,
        shared_extra: dict[str, Any],
        elapsed: float,
    ) -> AgentLoopOutput:
        resp_len = len(record.loss_mask)
        prompt_ids = record.token_ids[:-resp_len] if resp_len else list(record.token_ids)
        response_ids = record.token_ids[-resp_len:] if resp_len else []
        response_mask = list(record.loss_mask)
        response_logprobs = list(record.logprobs)
        assert len(response_ids) == len(response_mask) == len(response_logprobs)

        # verl pads prompts to prompt_length (left) and responses to
        # response_length (right); anything longer must be truncated here.
        prompt_ids = prompt_ids[-self.prompt_length :]
        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        response_logprobs = response_logprobs[: self.response_length]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=reward,
            num_turns=num_turns + 1,
            metrics=AgentLoopMetrics(generate_sequences=elapsed),
            extra_fields={
                **shared_extra,
                "trace_index": index,
                "trace_metadata": dict(record.metadata),
                # AgentLoopWorkerTQ reads this with brackets when broadcasting an
                # inline reward across multiple outputs — must always exist.
                "reward_extra_info": {},
            },
        )

    def _degenerate_output(
        self, kwargs: dict[str, Any], sid: str, error: str, elapsed: float, result: dict[str, Any] | None = None
    ) -> AgentLoopOutput:
        """A valid-but-inert output for failed rollouts. Never raise — a raised
        exception marks the whole prompt group as failed in the v1 replay buffer.

        The single pad-token response carries response_mask=[1] (NOT 0: verl's
        rollout-correction helper requires at least one valid response token per
        row) with rollout_log_probs=[0.0], and scores 0.0 in built_in mode."""
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        prompt_ids = [pad_id]
        raw_prompt = kwargs.get("raw_prompt")
        if raw_prompt is not None:
            try:
                prompt_ids = self.tokenizer.apply_chat_template(
                    list(raw_prompt), add_generation_prompt=True, tokenize=True
                )
                if hasattr(prompt_ids, "input_ids"):  # BatchEncoding
                    prompt_ids = prompt_ids.input_ids
                prompt_ids = list(prompt_ids)[-self.prompt_length :]
            except Exception:
                prompt_ids = [pad_id]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=[pad_id],
            response_mask=[1],
            response_logprobs=[0.0],
            # None is reserved for reward_mode="separate", which __init__ rejects today
            reward_score=0.0 if self.reward_mode == "built_in" else None,
            num_turns=1,
            metrics=AgentLoopMetrics(generate_sequences=elapsed),
            extra_fields={
                "acr_failed": True,
                "acr_error": error,
                "acr_result": result or {},
                "acr_session_id": sid,
                "num_trace_records": 0,
                "trace_index": 0,
                "reward_extra_info": {},
                # v1 staleness tags must be real ints (see run())
                "min_global_steps": int(kwargs.get("global_steps", 0)),
                "max_global_steps": int(kwargs.get("global_steps", 0)),
            },
        )


__all__ = ["AgentCoreAgentLoop"]
