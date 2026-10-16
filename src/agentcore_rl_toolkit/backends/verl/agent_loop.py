"""``AgentCoreAgentLoop`` — verl agent loop that runs rollouts on Bedrock
AgentCore Runtime with token capture through the in-repo rollout gateway.

Per rollout: create a gateway session keyed by a fresh uuid → invoke the ACR
agent with that uuid as the ACR ``runtimeSessionId`` (the container places it in
its LLM client's api-key slot, which the gateway reads as the Bearer sid) → await
the agent's background HTTP invocation result (including its scalar reward) →
drain the gateway session into TraceRecords → reshape each into an
``AgentLoopOutput``.

Registered as ``agentcore_agent`` — enable via verl's
``rollout.agent.agent_loop_config_path``. Requires ``trainer.use_v1=true``:
``run`` returns ``list[AgentLoopOutput]`` (one per trajectory-tree leaf, e.g.
sub-agent forks), which only the v1 TransferQueue path consumes.
"""

import asyncio
import logging
import math
import time
import uuid
from typing import Any

import ray
from pydantic import ValidationError
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister, PersistentDict
from agentcore_rl_toolkit.concurrency.ray_adapters import RayPrioritySemaphore, RayRateLimiter
from agentcore_rl_toolkit.rollout_gateway import BaseTrace, TraceRecord
from agentcore_rl_toolkit.rollout_session.errors import RolloutContractError
from agentcore_rl_toolkit.rollout_session.factory import make_session

from .gateway_host import GatewayHandle, get_or_start_gateway

logger = logging.getLogger(__name__)


def _extract_agent_reward(result: dict) -> float:
    """Reject missing/coerced rewards rather than silently flattening GRPO groups."""
    reward = result.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError(f"Expected a finite scalar agent reward, got {reward!r}")
    return float(reward)


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
        rollout_session_backend: dict[str, Any],
        max_tokens_per_turn: int,
        rollout_concurrency: int = 128,
        tps_limit: int = 5,
        max_rollout_time: float = 1800.0,
        gateway_bind_host: str = "0.0.0.0",
        gateway_port: int = 0,
        gateway_public_host: str | None = None,
        gateway_adapters: list[str] | None = None,
        max_turns_per_sid: int | None = None,
        history_mode: str = "tree",
        linear_on_nonlinear: str = "reset",
        reward_mode: str = "built_in",
        reward_extra_info_defaults: dict | None = None,
        # {name: threshold} -> emit reward_extra_info[name] = 1.0 if reward >= threshold.
        reward_thresholds: dict | None = None,
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
                "and have the agent return {'reward': ...} in its session result."
            )
        if reward_mode != "built_in":
            raise ValueError(f"reward_mode must be 'built_in', got {reward_mode!r}")

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_model_len = self.rollout_config.max_model_len
        self._validate_token_budgets(max_tokens_per_turn)
        self.max_tokens_per_turn = max_tokens_per_turn
        # The gateway cannot emit more tokens than verl's response region can store.
        # The two padded regions may sum past max_model_len; valid tokens never do.
        self.max_context_tokens = self.response_length
        self.max_rollout_time = max_rollout_time
        self.rollout_session_backend = dict(rollout_session_backend)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (rollout_concurrency, tps_limit)
        ):
            raise ValueError("rollout_concurrency and tps_limit must be positive integers")
        # Kept as config, not inlined, so a validated trainer-side mode can land
        # without a signature change. Reward semantics: see the README.
        self.reward_mode = reward_mode
        self._rei_defaults = {str(k): float(v) for k, v in (reward_extra_info_defaults or {}).items()}
        # verl already derives its `reward` validation metric from rm_scores.
        self._rei_defaults.pop("reward", None)
        self._reward_thresholds = {str(k): float(v) for k, v in (reward_thresholds or {}).items()}
        if collisions := sorted(set(self._rei_defaults) & set(self._reward_thresholds)):
            raise ValueError(
                f"reward_thresholds and reward_extra_info_defaults both declare {collisions}. "
                "A threshold is derived from the reward and would overwrite the agent-reported "
                "metric of the same name. Rename one side."
            )
        self.model_id = self.config.actor_rollout_ref.model.path

        self._gateway: GatewayHandle = get_or_start_gateway(
            server_manager=server_manager,
            tokenizer=tokenizer,
            host=gateway_bind_host,
            port=gateway_port,
            public_host=gateway_public_host,
            adapters=gateway_adapters,
            max_turns_per_sid=max_turns_per_sid,
            chat_template_kwargs=dict(self.apply_chat_template_kwargs or {}),
            history_mode=history_mode,
            linear_on_nonlinear=linear_on_nonlinear,
        )
        self._rollout_semaphore = RayPrioritySemaphore(ray.get_actor("rollout_concurrency_semaphore"))
        self._session_rate_limiter = RayRateLimiter(ray.get_actor("session_create_rate_limiter"))

    def _validate_token_budgets(self, max_tokens_per_turn: int) -> None:
        if self.max_model_len is None or self.max_model_len <= 0:
            raise ValueError(
                "AgentCoreAgentLoop requires an explicit positive "
                "actor_rollout_ref.rollout.max_model_len. This is the model context "
                "capacity; verl validates it against the Hugging Face model config."
            )
        if self.prompt_length > self.max_model_len:
            raise ValueError(
                f"rollout.prompt_length ({self.prompt_length}) cannot exceed "
                f"rollout.max_model_len ({self.max_model_len})"
            )
        if self.response_length > self.max_model_len:
            raise ValueError(
                f"rollout.response_length ({self.response_length}) cannot exceed "
                f"rollout.max_model_len ({self.max_model_len})"
            )
        if type(max_tokens_per_turn) is not int or max_tokens_per_turn <= 0:  # noqa: E721 - reject bool
            raise ValueError(f"max_tokens_per_turn must be a positive integer, got {max_tokens_per_turn!r}")
        if max_tokens_per_turn > self.max_model_len:
            raise ValueError(
                f"max_tokens_per_turn ({max_tokens_per_turn}) cannot exceed "
                f"rollout.max_model_len ({self.max_model_len})"
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
        task = {
            **payload,
            "_config": {
                "base_url": f"{self._gateway.base_url}/v1",
                "model_id": self.model_id,
                "api_key": sid,
            },
        }

        self._gateway.gateway.create_session(
            sid,
            sampling_defaults=self._sampling_defaults(sampling_params),
            max_context_tokens=self.max_context_tokens,
        )

        result: dict[str, Any] = {}
        error: str | None = None
        try:
            state = PersistentDict({"session_id": sid}, persister=NullPersister())
            session = make_session(sid, self.rollout_session_backend, state)
            async with self._rollout_semaphore.slot(), session:
                await self._session_rate_limiter.wait_async()
                async with asyncio.timeout(self.max_rollout_time):
                    await session.setup(task)
                    dump = await session.run(task)
            result = dump.task_output or {}
            error = dump.exception
        except (RolloutContractError, ValidationError):
            raise
        except asyncio.TimeoutError:
            error = f"rollout timed out after {self.max_rollout_time}s"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        if error:
            logger.warning("ACR rollout failed (sid=%s): %s", sid, error)

        num_turns = self._gateway.gateway.manager.turn_count(sid)
        records = await self._gateway.gateway.finish_session(sid, base_sample=BaseTrace(rollout_id=sid), reward=0.0)
        records = [r for r in records if r.token_ids]
        engine_extra = self._gateway.backend.pop_extra_fields(sid)
        elapsed = time.monotonic() - start

        if not records:
            self._warn_if_static_session_capture(sid)
            reason = error or "no model turns were captured"
            raise RuntimeError(f"ACR rollout produced no trainable trajectory (sid={sid}): {reason}")

        reward = self._resolve_reward(result, error, sid)

        # v1's staleness metrics do int(tag["min_global_steps"]) — the tags must
        # always be real ints. The engine's extra_fields carry them for turns it
        # served; default to the dataloader step otherwise.
        global_steps = int(kwargs.get("global_steps", 0))
        shared_extra = {
            # the session result is parsed JSON —
            # already plain python, no sanitizing needed
            "acr_result": result,
            "acr_session_id": sid,
            "metrics": {
                "acr_failed": float(error is not None),
                "num_trace_records": float(len(records)),
            },
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

        shared_extra["reward_extra_info"] = self._reward_extra_info(result, reward)

        outputs = [
            self._record_to_output(r, i, reward, num_turns, shared_extra, elapsed) for i, r in enumerate(records)
        ]
        return outputs

    def _reward_extra_info(self, result: dict[str, Any] | None, reward: float) -> dict:
        info: dict[str, float] = dict(self._rei_defaults)
        metrics = (result or {}).get("metrics") or {}
        if isinstance(metrics, dict):
            for k in self._rei_defaults:
                v = metrics.get(k)
                if isinstance(v, bool | int | float):
                    try:
                        info[k] = float(v)
                    except (TypeError, ValueError):
                        continue

        info["agent_reward"] = float(reward)
        for name, threshold in self._reward_thresholds.items():
            info[name] = 1.0 if float(reward) >= threshold else 0.0
        return info

    # -- helpers ---------------------------------------------------------------

    def _resolve_reward(self, result: dict[str, Any], error: str | None, sid: str) -> float:
        """The rollout's reward_score, which becomes rm_scores directly (verl
        skips reward computation for this rollout). The agent owns scoring;
        failures with a partial trajectory score 0."""
        if error is not None:
            return 0.0
        return _extract_agent_reward(result)

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
                    "api_key instead of _config.api_key (stale agent image?). "
                    "See the agent-side contract in backends/verl/README.md.",
                    sid,
                    static_sid,
                )
                break

    def _sampling_defaults(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        defaults: dict[str, Any] = {"max_new_tokens": self.max_tokens_per_turn}
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
            if "_config" in payload or "_agentcore_runtime" in payload:
                raise ValueError("Dataset payload must not contain reserved fields '_config' or '_agentcore_runtime'")
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
        response_region_len = len(record.loss_mask)
        prompt_end = len(record.token_ids) - response_region_len
        initial_prompt = list(record.token_ids[:prompt_end])
        response_region_ids = list(record.token_ids[prompt_end:])
        response_mask = list(record.loss_mask)
        response_logprobs = list(record.logprobs)

        # verl stores prompts and responses in fixed-width regions. Preserve the
        # complete token order when the initial prompt exceeds its region by placing
        # the overflow at the start of the response region. These are input tokens:
        # they remain attended to, but carry no policy loss or rollout logprob.
        prompt_ids = initial_prompt[: self.prompt_length]
        prompt_overflow = initial_prompt[self.prompt_length :]
        response_ids = prompt_overflow + response_region_ids
        response_mask = [0] * len(prompt_overflow) + response_mask
        response_logprobs = [0.0] * len(prompt_overflow) + response_logprobs

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
                "reward_extra_info": dict(shared_extra.get("reward_extra_info") or {}),
            },
        )


__all__ = ["AgentCoreAgentLoop"]
