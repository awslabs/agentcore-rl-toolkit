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
        max_pool_connections: 100        # boto3 conn-pool size (>= max_concurrent)
        reward_postprocessing: "grpo"    # "grpo" or "identity"
"""

import asyncio
import copy
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

# Cache of eval datasets keyed by EvalDatasetConfig.cache_key, so repeated
# evaluations don't re-read + re-tokenize the same JSONL every rollout.
_eval_datasets: dict = {}


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
    max_pool_connections: int = 10
    reward_postprocessing: str = "grpo"
    cumulative_token_mode: bool = False
    renderer_family: str = "auto"
    sglang_tool_call_parser: str | None = None
    sglang_reasoning_parser: str | None = None
    gateway_log_level: str = "warning"

    @classmethod
    def from_args(cls, args: Namespace) -> "SlimeArtConfig":
        """Build config from slime args, falling back to env vars then defaults."""

        def _get(attr: str, env: str, default):
            val = getattr(args, attr, None)
            if val is not None and val != "" and val != default:
                return val
            return os.environ.get(env, default)

        def _as_bool(val) -> bool:
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in ("1", "true", "yes", "on")

        return cls(
            agent_runtime_arn=_get("agent_runtime_arn", "ACR_AGENT_RUNTIME_ARN", cls.agent_runtime_arn),
            s3_bucket=_get("s3_bucket", "ACR_S3_BUCKET", cls.s3_bucket),
            exp_id=_get("exp_id", "EXP_ID", cls.exp_id),
            gateway_port=int(_get("gateway_port", "GATEWAY_PORT", cls.gateway_port)),
            acr_timeout=float(_get("acr_timeout", "ACR_TIMEOUT", cls.acr_timeout)),
            model_id=_get("model_id", "MODEL_ID", cls.model_id),
            acr_tps_limit=int(_get("acr_tps_limit", "ACR_TPS_LIMIT", cls.acr_tps_limit)),
            max_concurrent=int(_get("max_concurrent", "MAX_CONCURRENT", cls.max_concurrent)),
            max_pool_connections=int(_get("max_pool_connections", "MAX_POOL_CONNECTIONS", cls.max_pool_connections)),
            reward_postprocessing=_get("reward_postprocessing", "REWARD_POSTPROCESSING", cls.reward_postprocessing),
            cumulative_token_mode=_as_bool(
                _get("cumulative_token_mode", "CUMULATIVE_TOKEN_MODE", cls.cumulative_token_mode)
            ),
            renderer_family=_get("renderer_family", "RENDERER_FAMILY", cls.renderer_family),
            # Read slime's own SGLang server args (args.sglang_tool_call_parser /
            # args.sglang_reasoning_parser) so the gateway parses identically.
            sglang_tool_call_parser=_get(
                "sglang_tool_call_parser", "SGLANG_TOOL_CALL_PARSER", cls.sglang_tool_call_parser
            ),
            sglang_reasoning_parser=_get(
                "sglang_reasoning_parser", "SGLANG_REASONING_PARSER", cls.sglang_reasoning_parser
            ),
            gateway_log_level=_get("gateway_log_level", "GATEWAY_LOG_LEVEL", cls.gateway_log_level),
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
        # The slime backend always drives the gateway in use_sglang mode (enforced
        # unconditionally in SlimeGatewayManager.start): the served engine is SGLang
        # behind sgl-router, whose token ids + logprobs are only exposed via the
        # native /generate API — the OpenAI endpoints would capture no token ids.
        # `model` (= slime's --hf-checkpoint) is the tokenizer the gateway renders
        # prompts to token ids with, and is required.
        _gateway = SlimeGatewayManager(
            GatewayConfig(
                port=cfg.gateway_port,
                host=args.sglang_router_ip,  # Bind to routable IP so VPC agents can reach it
                strip_vllm_fields=False,
                cumulative_token_mode=cfg.cumulative_token_mode,
                renderer_family=cfg.renderer_family,
                model=getattr(args, "hf_checkpoint", None),
                sglang_tool_call_parser=cfg.sglang_tool_call_parser,
                sglang_reasoning_parser=cfg.sglang_reasoning_parser,
                log_level=cfg.gateway_log_level,
            )
        )
        _gateway.start(sglang_url)

    if _client is None:
        _client = RolloutClient(
            agent_runtime_arn=cfg.agent_runtime_arn,
            s3_bucket=cfg.s3_bucket,
            exp_id=cfg.exp_id,
            tps_limit=cfg.acr_tps_limit,
            max_pool_connections=cfg.max_pool_connections,
        )

    return _gateway, _client, cfg


# ---------------------------------------------------------------------------
# Payload conversion
# ---------------------------------------------------------------------------


def _sample_to_payload(sample) -> dict:
    """The agent payload is the JSONL row's ``metadata`` dict, verbatim.

    slime's Dataset reads the JSONL row's ``metadata`` field into
    ``Sample.metadata``; we hand that dict to the agent unchanged. The JSONL's
    top-level ``prompt`` field is for slime (tokenization, length filtering);
    the agent's payload shape is entirely defined by whatever the data author
    put in ``metadata``. A shallow copy isolates the agent's view from
    downstream mutations to ``Sample.metadata`` (e.g. ``task_metadata``
    injection in ``_process_one_episode``).
    """
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


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
    from .traces import extract_reward, merge_traces_to_samples

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
        _log_traces(session_id, traces, episode_reward, task_index)

        if not traces:
            noop = _make_noop_sample(group_index=task_index, session_id=session_id, status_name="FAILED")
            noop.metadata["episode_error"] = "no traces returned"
            logger.info("Episode failed (session=%s): %s", session_id, noop.metadata["episode_error"])
            return [noop]

        # Always merge: fold the trajectory into one Sample per contiguous
        # prefix-extending segment (GPU-efficient, drift-free). This is correct
        # regardless of transport (use_sglang) or prompt construction
        # (cumulative_token_mode): when turns share a byte-exact prefix they merge
        # into one sequence; when a turn breaks the prefix (e.g. non-cumulative
        # full re-render, or agent context surgery) merge_traces_to_samples splits
        # at that point, degrading to per-turn/per-segment Samples.
        samples = merge_traces_to_samples(
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
        # Record the failure on the sample (the per-batch summary counts these)
        # and log it at INFO so a failing episode is visible.
        noop = _make_noop_sample(group_index=task_index, session_id=session_id, status_name="FAILED")
        noop.metadata["episode_error"] = str(e) or type(e).__name__
        logger.info("Episode failed (session=%s): %s", session_id, noop.metadata["episode_error"])
        return [noop]
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

    # ---- Step 1: resolve the prompt groups to run, each paired with the
    # sampling params it should use ----
    # Training samples one batch of GRPO-grouped prompts from the live
    # data_source, all sharing the rollout params. Eval reads held-out datasets
    # (args.eval_datasets, built by slime from --eval-prompt-data /
    # --eval-config), each carrying its own already-resolved params (e.g.
    # greedy via --eval-temperature 0); eval rewards are independent, so each
    # (prompt, sample) is its own size-1 group.
    if evaluation:
        groups = []  # list of (prompt_group, sampling_params)
        for dataset_cfg in getattr(args, "eval_datasets", None) or []:
            params = {
                "temperature": dataset_cfg.temperature,
                "top_p": dataset_cfg.top_p,
                "top_k": dataset_cfg.top_k,
                "max_new_tokens": dataset_cfg.max_response_len,
            }
            dataset = _get_eval_dataset(args, dataset_cfg)
            for prompt in dataset.samples:
                for _ in range(dataset_cfg.n_samples_per_eval_prompt or 1):
                    groups.append(([copy.deepcopy(prompt)], params))
    else:
        params = {
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
            "top_k": args.rollout_top_k,
            "max_new_tokens": args.rollout_max_response_len,
        }
        groups = [(group, params) for group in data_source.get_samples(args.rollout_batch_size)]

    # ---- Step 2 (shared): run every group as parallel ACR episodes ----
    # Each sample in a group becomes one episode tagged with the group index
    # (GRPO grouping in training, a unique id in eval); turns are flattened
    # back per group. All groups (and all episodes within them) are scheduled
    # concurrently, but a shared semaphore caps the number of episodes that are
    # actually in flight at once (cfg.max_concurrent). Without this cap, a large
    # batch (e.g. a full 1319-prompt eval set) launches every episode at once —
    # the 25-TPS client limiter only paces session *starts*, not the live count —
    # which saturates the gateway/router + S3 result polling (episodes then miss
    # acr_timeout and fail) and over-pressures the colocated SGLang KV cache
    # (token-pool exhaustion crash). asyncio.gather preserves argument order, so
    # the returned list stays group-ordered (list[list[Sample]]), keeping the
    # GRPO group_index tags and slime's nesting-depth contract intact. (Ordering
    # is non-semantic anyway: grouping is by explicit group_index/session_id, not
    # list position — see rewards.normalize_episode_rewards.)
    sample_counter = iter(range(10**9))

    async def _run():
        # Bound concurrent in-flight episodes across ALL groups. Created inside
        # the running loop (asyncio.Semaphore binds to the active event loop).
        sem = asyncio.Semaphore(max(1, cfg.max_concurrent))

        async def _episode(s, group_index, sampling_params):
            async with sem:
                return await _process_one_episode(s, gateway, client, cfg, sampling_params, group_index, sample_counter)

        async def _run_group(group_index, group, sampling_params):
            results = await asyncio.gather(*(_episode(s, group_index, sampling_params) for s in group))
            return [s for r in results for s in r]

        return list(
            await asyncio.gather(
                *(
                    _run_group(group_index, group, sampling_params)
                    for group_index, (group, sampling_params) in enumerate(groups)
                )
            )
        )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    results = loop.run_until_complete(_run())

    # ---- Step 3: per-batch summary (replaces per-session logging) ----
    # num_episodes is the input-prompt count; num_sequences is the Sample
    # count after merge (multi-turn episodes fold toward one sequence each in
    # cumulative mode, stay one-per-turn otherwise).
    num_episodes = sum(len(group) for group, _ in groups)
    num_sequences = sum(len(g) for g in results)
    failed = sum(1 for g in results for s in g if (s.metadata or {}).get("episode_error"))
    succeeded = num_episodes - failed
    phase = "Eval" if evaluation else "Rollout"
    logger.info(
        "%s %d batch: episodes=%d (succeeded=%d failed=%d) sequences=%d",
        phase,
        rollout_id,
        num_episodes,
        succeeded,
        failed,
        num_sequences,
    )

    # ---- Step 4: shape the backend-specific output ----
    if evaluation:
        # Episode reward is broadcast to every turn-Sample; take the first.
        rewards = [float(g[0].reward) if g and isinstance(g[0].reward, (int, float)) else 0.0 for g in results]
        n = max(len(rewards), 1)
        accuracy = sum(1 for r in rewards if r > 0) / n
        avg_reward = sum(rewards) / n
        return RolloutFnEvalOutput(
            data={"eval": {"rewards": rewards}},
            metrics={
                "eval/accuracy": accuracy,
                "eval/avg_reward": avg_reward,
                "eval/n_samples": len(rewards),
            },
        )

    # Training: pad to a dp_size multiple so no real samples are trimmed.
    dp_size = args.actor_num_nodes * args.actor_num_gpus_per_node // args.tensor_model_parallel_size
    remainder = sum(len(g) for g in results) % dp_size
    if remainder > 0:
        results[-1].extend([_make_noop_sample(group_index=-1) for _ in range(dp_size - remainder)])
    return RolloutFnTrainOutput(samples=results)


def _get_eval_dataset(args, dataset_cfg):
    """Load + cache a held-out eval dataset described by an EvalDatasetConfig.

    Reads the JSONL itself (independent of the training data_source) using
    slime's Dataset so the prompt/metadata parsing matches the training path.
    """
    from slime.utils.data import Dataset
    from slime.utils.processing_utils import load_processor, load_tokenizer

    key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if key not in _eval_datasets:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        _eval_datasets[key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            metadata_key=dataset_cfg.metadata_key,
            multimodal_keys=args.multimodal_keys,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    return _eval_datasets[key]
