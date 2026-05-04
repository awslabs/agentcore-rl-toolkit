"""Custom slime rollout function using ACR agents + rllm-model-gateway.

Each LLM call becomes a separate slime Sample. All Samples from the same
task share group_index for GRPO normalization. Episode rewards are broadcast
to all turns.

Usage:
    python -m slime.train \
        --rollout-function-path \
            agentcore_rl_toolkit.backends.slime.integration.rollout.generate_rollout \
        --custom-reward-post-process-path \
            agentcore_rl_toolkit.backends.slime.integration.rewards.normalize_episode_rewards \
        --custom-config-path config.yaml \
        --use-dynamic-global-batch-size \
        --use-dynamic-batch-size \
        --max-tokens-per-gpu 9216 \
        ...

    Configuration via --custom-config-path YAML:
        agent_runtime_arn: "arn:aws:bedrock-agentcore:..."
        s3_bucket: "my-bucket"
        exp_id: "slime-training"
        gateway_port: 9090               # rllm-model-gateway port
        acr_timeout: 900                 # per-session ACR invocation timeout
        model_id: "default"              # OpenAI model id served to the agent
        acr_tps_limit: 25                # ACR service TPS quota
        max_concurrent: 100              # max concurrent ACR sessions (eval)
        reward_postprocessing: "grpo"    # "grpo" or "identity"
"""

import asyncio
import json
import logging
import os
import uuid
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# File-based trace logging for debugging prompt/response content
_TRACE_LOG_PATH = Path(os.environ.get("TRACE_LOG", "trace_log.jsonl"))

# Module-level singletons (initialized on first call, reused across rollout steps)
_gateway = None
_client = None
_config = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SlimeArtConfig:
    """Configuration for ACR-based rollouts with slime.

    All fields come from slime's args namespace via --custom-config-path YAML
    (see module docstring). Env var fallbacks are provided as an override path
    for CI/dev convenience.
    """

    agent_runtime_arn: str = ""
    s3_bucket: str = ""
    exp_id: str = "slime-training"
    gateway_port: int = 9090
    acr_timeout: float = 900.0
    model_id: str = "default"
    acr_tps_limit: int = 25
    max_concurrent: int = 100
    reward_postprocessing: str = "grpo"

    @classmethod
    def from_args(cls, args: Namespace) -> "SlimeArtConfig":
        """Build config from slime args, falling back to env vars then defaults."""

        def _get(attr: str, env: str, default):
            val = getattr(args, attr, None)
            if val is not None and val != "" and val != default:
                return val
            return os.environ.get(env, default)

        return cls(
            agent_runtime_arn=_get("agent_runtime_arn", "ACR_AGENT_RUNTIME_ARN", cls.agent_runtime_arn),
            s3_bucket=_get("s3_bucket", "ACR_S3_BUCKET", cls.s3_bucket),
            exp_id=_get("exp_id", "EXP_ID", cls.exp_id),
            gateway_port=int(_get("gateway_port", "GATEWAY_PORT", cls.gateway_port)),
            acr_timeout=float(_get("acr_timeout", "ACR_TIMEOUT", cls.acr_timeout)),
            model_id=_get("model_id", "MODEL_ID", cls.model_id),
            acr_tps_limit=int(_get("acr_tps_limit", "ACR_TPS_LIMIT", cls.acr_tps_limit)),
            max_concurrent=int(_get("max_concurrent", "MAX_CONCURRENT", cls.max_concurrent)),
            reward_postprocessing=_get("reward_postprocessing", "REWARD_POSTPROCESSING", cls.reward_postprocessing),
        )


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _import_slime_types():
    try:
        from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
        from slime.utils.types import Sample

        return Sample, RolloutFnTrainOutput, RolloutFnEvalOutput
    except ImportError as err:
        raise ImportError(
            "slime is required for this module. Install with: pip install agentcore-rl-toolkit[slime]"
        ) from err


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_traces(session_id: str, traces: list, reward: float, task_index: int):
    """Append trace messages and responses to a JSONL file for debugging."""
    try:
        with open(_TRACE_LOG_PATH, "a") as f:
            for i, t in enumerate(traces):
                record = {
                    "session_id": session_id,
                    "task_index": task_index,
                    "turn": i,
                    "reward": reward,
                    "model": getattr(t, "model", ""),
                    "messages": getattr(t, "messages", []),
                    "response": getattr(t, "response_message", {}),
                    "finish_reason": getattr(t, "finish_reason", None),
                    "prompt_tokens": len(getattr(t, "prompt_token_ids", []) or []),
                    "completion_tokens": len(getattr(t, "completion_token_ids", []) or []),
                }
                f.write(json.dumps(record) + "\n")
    except Exception:
        logger.warning("Failed to write trace log", exc_info=True)


def _make_noop_sample(group_index: int = -1, session_id: str = "", status_name: str = "COMPLETED"):
    """Create a minimum-valid Sample that contributes zero gradient.

    Used for DP padding and failed episodes. Has 2 tokens (1 prompt + 1 response),
    loss_mask=[0] so Megatron processes it without error but produces no gradient.

    For failed episodes, pass session_id so that normalize_episode_rewards counts
    this as a separate episode (reward=0) in the GRPO group.
    """
    Sample, _, _ = _import_slime_types()
    s = Sample()
    s.tokens = [0, 0]
    s.response_length = 1
    s.loss_mask = [0]
    s.reward = 0.0
    s.rollout_log_probs = [0.0]
    s.group_index = group_index
    s.status = Sample.Status[status_name]
    if session_id:
        s.session_id = session_id
        s.metadata = {"gateway_session_id": session_id, "task_index": group_index, "turn_index": 0}
    return s


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _ensure_initialized(args: Namespace):
    """Lazily initialize gateway, client, and tokenizer on first rollout call."""
    global _gateway, _client, _config, _tokenizer

    from agentcore_rl_toolkit import RolloutClient

    from .gateway import GatewayConfig, SlimeGatewayManager

    if _config is None:
        _config = SlimeArtConfig.from_args(args)

    cfg = _config

    if _gateway is None:
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        _gateway = SlimeGatewayManager(
            GatewayConfig(
                port=cfg.gateway_port,
                host=args.sglang_router_ip,  # Bind to routable IP so VPC agents can reach it
                strip_vllm_fields=False,
            )
        )
        _gateway.start(sglang_url)

    if _client is None:
        _client = RolloutClient(
            agent_runtime_arn=cfg.agent_runtime_arn,
            s3_bucket=cfg.s3_bucket,
            exp_id=cfg.exp_id,
            tps_limit=cfg.acr_tps_limit,
        )

    return _gateway, _client, cfg


# ---------------------------------------------------------------------------
# Payload conversion
# ---------------------------------------------------------------------------


def _sample_to_payload(sample) -> dict:
    """Convert a slime Sample to an ART invocation payload.

    Extracts all non-None public fields from the Sample into the payload.
    The agent's @rollout_entrypoint receives this dict as `payload`.
    """
    payload = {}

    if hasattr(sample, "prompt") and sample.prompt:
        payload["prompt"] = sample.prompt
    if hasattr(sample, "label") and sample.label is not None:
        payload["answer"] = sample.label
    if hasattr(sample, "metadata") and sample.metadata:
        payload["metadata"] = sample.metadata

    if hasattr(sample, "to_dict"):
        for key, value in sample.to_dict().items():
            if (
                key not in payload
                and value is not None
                and key
                not in (
                    "tokens",
                    "rollout_log_probs",
                    "loss_mask",
                    "teacher_log_probs",
                    "rollout_routed_experts",
                    "multimodal_inputs",
                    "multimodal_train_inputs",
                    "group_index",
                    "index",
                    "status",
                    "session_id",
                    "spec_info",
                    "prefix_cache_info",
                    "response_length",
                    "response",
                    "weight_versions",
                    "remove_sample",
                    "non_generation_time",
                    "generate_function_path",
                    "train_metadata",
                )
            ):
                payload[key] = value

    return payload


# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------


async def _process_one_episode(
    sample,
    gateway,
    client,
    cfg,
    sampling_params: dict,
    task_index: int,
    sample_counter,
) -> list:
    """Run one agent episode, return per-turn Samples.

    All returned Samples share task_index (as group_index) so that
    normalize_episode_rewards() can group all turns from all episodes
    of the same task together for GRPO normalization.
    """
    from .traces import episode_traces_to_samples, extract_reward

    session_id = str(uuid.uuid4())
    try:
        session_url = await asyncio.to_thread(gateway.create_session, session_id, sampling_params)

        payload = _sample_to_payload(sample)
        # Translate to OpenAI-compatible params (max_new_tokens→max_tokens, drop top_k)
        agent_params = {
            k if k != "max_new_tokens" else "max_tokens": v for k, v in sampling_params.items() if k != "top_k"
        }
        future = await client.invoke_async(
            payload=payload,
            session_id=session_id,
            input_id=session_id,
            base_url=session_url,
            model_id=cfg.model_id,
            sampling_params=agent_params,
        )

        result = await future.result_async(timeout=cfg.acr_timeout)
        traces = await asyncio.to_thread(gateway.get_traces, session_id)
        episode_reward = extract_reward(result)
        logger.info(
            "Session %s: status=%s, traces=%d, reward=%s",
            session_id[:8],
            result.get("status_code"),
            len(traces),
            episode_reward,
        )
        _log_traces(session_id, traces, episode_reward, task_index)

        if not traces:
            return [_make_noop_sample(group_index=task_index, session_id=session_id, status_name="FAILED")]

        samples = episode_traces_to_samples(
            traces=traces,
            episode_reward=episode_reward,
            group_index=task_index,
            session_id=session_id,
            sample_counter=sample_counter,
        )

        for s in samples:
            s.prompt = sample.prompt
            s.label = sample.label
            if sample.metadata:
                s.metadata["task_metadata"] = sample.metadata

        return samples

    except Exception as e:
        logger.error("Episode failed (session=%s): %s", session_id, e)
        return [_make_noop_sample(group_index=task_index, session_id=session_id, status_name="FAILED")]
    finally:
        await asyncio.to_thread(gateway.delete_session, session_id)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source,
    evaluation: bool = False,
):
    """Custom slime rollout function: ACR agents + gateway.

    Implements slime's --rollout-function-path interface.
    """
    _, RolloutFnTrainOutput, RolloutFnEvalOutput = _import_slime_types()
    gateway, client, cfg = _ensure_initialized(args)

    batch_size = (
        args.rollout_batch_size if not evaluation else getattr(args, "eval_batch_size", args.rollout_batch_size)
    )
    sample_groups = data_source.get_samples(batch_size)

    sampling_params = {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "top_k": args.rollout_top_k,
        "max_new_tokens": args.rollout_max_response_len,
    }

    sample_counter = iter(range(10**9))

    async def _run():
        all_groups = []
        for task_index, task_episodes in enumerate(sample_groups):
            episode_tasks = [
                _process_one_episode(s, gateway, client, cfg, sampling_params, task_index, sample_counter)
                for s in task_episodes
            ]
            episode_results = await asyncio.gather(*episode_tasks)
            flat = [s for result in episode_results for s in result]
            all_groups.append(flat)
        return all_groups

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    results = loop.run_until_complete(_run())

    if evaluation:
        return _build_eval_output(results, RolloutFnEvalOutput)

    # Pad to dp_size multiple so no real samples are trimmed
    dp_size = args.actor_num_nodes * args.actor_num_gpus_per_node // args.tensor_model_parallel_size
    flat_count = sum(len(g) for g in results)
    remainder = flat_count % dp_size
    if remainder > 0:
        pad_count = dp_size - remainder
        results[-1].extend([_make_noop_sample(group_index=-1) for _ in range(pad_count)])

    return RolloutFnTrainOutput(samples=results)


def _build_eval_output(results, RolloutFnEvalOutput):
    """Convert grouped samples into slime eval output format."""
    all_rewards = []
    for group in results:
        for sample in group:
            r = sample.reward if sample.reward is not None else 0.0
            all_rewards.append(float(r) if isinstance(r, (int, float)) else 0.0)

    accuracy = sum(1 for r in all_rewards if r > 0) / max(len(all_rewards), 1)
    return RolloutFnEvalOutput(
        data={"eval": {"rewards": all_rewards}},
        metrics={"eval/accuracy": accuracy, "eval/n_samples": len(all_rewards)},
    )
