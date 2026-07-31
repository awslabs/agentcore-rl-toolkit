"""Custom slime rollout function using ACR agents + the in-repo rollout gateway.

:class:`agentcore_rl_toolkit.rollout_gateway.RolloutGateway` is served in-process
(an aiohttp app in a dedicated thread) and owns tokenization: it renders each turn
with the HF chat template of ``--hf-checkpoint``, samples token-in/token-out via
SGLang's native ``/generate`` (``SglangHttpBackend``), and linearizes each session's
message tree into loss-masked :class:`TraceRecord` rows — multi-turn prefix merging
happens inside the gateway, so this module only converts records to slime Samples.

Session identity — api-key / Bearer slot: the gateway keys sessions off the
api-key slot of the agent's LLM client (what OpenAI/Anthropic SDKs — and harnesses
like Claude Code / Codex — forward on every request). Each episode sends its
session id as ``_rollout["api_key"]``; the rl_app plugs it into its model client
(``api_key=payload["_rollout"].get("api_key", "EMPTY")``), so every LLM call
arrives at the fixed gateway ``base_url`` tagged ``Authorization: Bearer <sid>``.
No per-session URLs and no model wrapper are needed.

Usage:
    python -m slime.train \
        --rollout-function-path \
            agentcore_rl_toolkit.backends.experimental.slime.integration.rollout.generate_rollout \
        --custom-reward-post-process-path \
            agentcore_rl_toolkit.backends.experimental.slime.integration.rewards.normalize_episode_rewards \
        --custom-config-path config.yaml \
        --use-dynamic-batch-size \
        --max-tokens-per-gpu 9216 \
        ...

    Slime's own --sglang-tool-call-parser / --sglang-reasoning-parser args are
    honored: when the tool-call parser is set, the gateway derenders model output
    with SGLang's own detectors (see sglang_parsing.py; the names must match the
    served model); when unset, the gateway's dependency-free built-in parsing is
    used.

    Configuration via --custom-config-path YAML:
        agent_runtime_arn: "arn:aws:bedrock-agentcore:..."
        s3_bucket: "my-bucket"
        exp_id: "slime-training"
        gateway_port: 9090               # in-process rollout gateway port
        acr_timeout: 900                 # per-session ACR invocation timeout
        model_id: "default"              # OpenAI model id served to the agent
        acr_tps_limit: 25                # ACR service TPS quota
        max_concurrent: 100              # max concurrent ACR sessions
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

from agentcore_rl_toolkit.rollout_gateway import BaseTrace, Status

logger = logging.getLogger(__name__)

# File-based trace logging for debugging captured trajectories
_TRACE_LOG_PATH = Path(os.environ.get("TRACE_LOG", "trace_log.jsonl"))

# Module-level singletons (initialized on first call, reused across rollout steps)
_gateway_server = None
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
    sglang_tool_call_parser: str | None = None
    sglang_reasoning_parser: str | None = None

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
            max_pool_connections=int(_get("max_pool_connections", "MAX_POOL_CONNECTIONS", cls.max_pool_connections)),
            reward_postprocessing=_get("reward_postprocessing", "REWARD_POSTPROCESSING", cls.reward_postprocessing),
            # Read slime's own SGLang server args (args.sglang_tool_call_parser /
            # args.sglang_reasoning_parser) so the gateway parses identically.
            sglang_tool_call_parser=_get(
                "sglang_tool_call_parser", "SGLANG_TOOL_CALL_PARSER", cls.sglang_tool_call_parser
            ),
            sglang_reasoning_parser=_get(
                "sglang_reasoning_parser", "SGLANG_REASONING_PARSER", cls.sglang_reasoning_parser
            ),
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
# Gateway assembly (slime-specific: SGLang backend + --hf-checkpoint template)
# ---------------------------------------------------------------------------


def _start_gateway_server(
    *,
    host: str,
    port: int,
    sglang_url: str,
    hf_checkpoint: str,
    acr_timeout: float,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
):
    """Assemble the slime-flavored gateway and serve it on a background thread.

    Slime always drives SGLang, so the gateway samples via ``SglangHttpBackend``
    and renders with the HF chat template of ``--hf-checkpoint``. The serving
    mechanics live in the shared :class:`ThreadedGatewayServer`.

    ``tool_call_parser`` / ``reasoning_parser`` (slime's --sglang-tool-call-parser /
    --sglang-reasoning-parser) select SGLang's engine-grade parsers for the served
    model's output format; the gateway samples via the native /generate endpoint,
    so that parsing happens here rather than in the SGLang server. Each is
    independent: an unset one leaves that derender stage on the gateway's
    dependency-free default.
    """
    try:
        from transformers import AutoTokenizer

        from agentcore_rl_toolkit.rollout_gateway import HfTemplateRenderer, RolloutGateway, ThreadedGatewayServer
        from agentcore_rl_toolkit.rollout_gateway.sampling_backends.sglang_http import SglangHttpBackend
    except ImportError as err:
        raise ImportError(
            "The rollout gateway requires aiohttp + transformers. "
            "Install with: pip install agentcore-rl-toolkit[gateway]"
        ) from err

    # Each stage falls back to the gateway's dependency-free default when its
    # slime arg is unset, so the two can be configured independently.
    parser_fns = {}
    if tool_call_parser:
        from .sglang_parsing import build_tool_parser

        parser_fns["tool_parser"] = build_tool_parser(tool_call_parser)
    if reasoning_parser:
        from .sglang_parsing import build_reasoning_parser

        parser_fns["reasoning_parser"] = build_reasoning_parser(reasoning_parser)

    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
    gateway = RolloutGateway(
        backend=SglangHttpBackend(sglang_url, sock_read_timeout=acr_timeout),
        renderer=HfTemplateRenderer(tokenizer, **parser_fns),
        tokenizer=tokenizer,
    )
    server = ThreadedGatewayServer(gateway, host=host, port=port)
    server.start()
    logger.info("Rollout gateway serving at %s (sglang worker: %s)", server.base_url, sglang_url)
    return server


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _ensure_initialized(args: Namespace):
    """Lazily initialize the in-process gateway and ACR client on first rollout call."""
    global _gateway_server, _client, _config

    from agentcore_rl_toolkit import RolloutClient

    if _config is None:
        _config = SlimeArtConfig.from_args(args)

    cfg = _config

    if _gateway_server is None:
        hf_checkpoint = getattr(args, "hf_checkpoint", None)
        if not hf_checkpoint:
            raise ValueError(
                "--hf-checkpoint is required: the rollout gateway renders prompts "
                "with the served checkpoint's HF chat template."
            )
        _gateway_server = _start_gateway_server(
            host=args.sglang_router_ip,  # bind to routable IP so VPC agents can reach it
            port=cfg.gateway_port,
            sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            hf_checkpoint=hf_checkpoint,
            acr_timeout=cfg.acr_timeout,
            tool_call_parser=cfg.sglang_tool_call_parser,
            reasoning_parser=cfg.sglang_reasoning_parser,
        )

    if _client is None:
        _client = RolloutClient(
            agent_runtime_arn=cfg.agent_runtime_arn,
            s3_bucket=cfg.s3_bucket,
            exp_id=cfg.exp_id,
            tps_limit=cfg.acr_tps_limit,
            max_pool_connections=cfg.max_pool_connections,
        )

    return _gateway_server, _client, cfg


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


def _extract_reward(acr_result: dict) -> float:
    """Extract scalar reward from an ACR S3 result dict."""
    rewards = acr_result.get("rewards", 0.0)
    if isinstance(rewards, list):
        return rewards[-1] if rewards else 0.0
    return float(rewards)


# ---------------------------------------------------------------------------
# TraceRecord -> slime Sample conversion
# ---------------------------------------------------------------------------


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
        s.metadata = {"gateway_session_id": session_id, "task_index": group_index, "record_index": 0}
    return s


def _record_to_sample(
    record,
    group_index: int,
    sample_index: int,
    session_id: str,
    record_index: int,
):
    """Convert one gateway TraceRecord into a slime Sample.

    A TraceRecord is already a merged, loss-masked training row: ``token_ids`` is
    the full sequence, ``loss_mask`` / ``logprobs`` cover the response region only
    (bridge tokens between turns carry loss_mask=0), so the conversion is direct.
    """
    Sample = _import_slime_types()[0]

    tokens = list(record.token_ids)
    loss_mask = list(record.loss_mask)
    logprobs = list(record.logprobs)
    response_length = int(record.response_length or len(loss_mask))

    # Megatron requires prompt_length >= 1; a record whose first-turn prompt is
    # empty cannot occur with a chat template, but guard anyway.
    if len(tokens) - response_length < 1:
        tokens = [0] + tokens

    # Defensive alignment: mask/logprobs must both span the response region.
    if len(loss_mask) != response_length:
        loss_mask = (loss_mask + [0] * response_length)[:response_length]
    if len(logprobs) != response_length:
        logprobs = (logprobs + [0.0] * response_length)[:response_length]

    sample = Sample()
    sample.tokens = tokens
    sample.response_length = response_length
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = logprobs
    sample.reward = float(record.reward)
    sample.group_index = group_index
    sample.index = sample_index
    sample.session_id = session_id
    sample.metadata = {
        **(record.metadata or {}),
        "task_index": group_index,
        "gateway_session_id": session_id,
        "record_index": record_index,
    }

    truncated = record.status is Status.TRUNCATED or (record.metadata or {}).get("truncated")
    sample.status = Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED
    return sample


def _log_records(session_id: str, records: list, reward: float, task_index: int):
    """Append captured TraceRecords to a JSONL file for debugging."""
    try:
        with open(_TRACE_LOG_PATH, "a") as f:
            for i, r in enumerate(records):
                record = {
                    "session_id": session_id,
                    "task_index": task_index,
                    "record": i,
                    "reward": reward,
                    "rollout_id": r.rollout_id,
                    "status": r.status.value,
                    "total_tokens": len(r.token_ids),
                    "response_length": r.response_length,
                    "trained_tokens": sum(r.loss_mask),
                    "response": r.response,
                    "metadata": r.metadata,
                }
                f.write(json.dumps(record) + "\n")
    except Exception:
        logger.warning("Failed to write trace log", exc_info=True)


# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------


def _session_sampling_defaults(sampling_params: dict) -> dict:
    """Per-session sampling defaults for the gateway (canonical keys, None-filtered)."""
    return {k: v for k, v in sampling_params.items() if v is not None}


async def _process_one_episode(
    sample,
    server,
    client,
    cfg,
    sampling_params: dict,
    task_index: int,
    sample_counter,
) -> list:
    """Run one agent episode, return its slime Samples.

    All returned Samples share task_index (as group_index) so that
    normalize_episode_rewards() can group all rows from all episodes
    of the same task together for GRPO normalization.
    """
    gateway = server.gateway
    session_id = str(uuid.uuid4())
    try:
        gateway.create_session(session_id, sampling_defaults=_session_sampling_defaults(sampling_params))

        payload = _sample_to_payload(sample)
        # Translate to OpenAI-compatible params (max_new_tokens→max_tokens, drop top_k)
        agent_params = {
            k if k != "max_new_tokens" else "max_tokens": v for k, v in sampling_params.items() if k != "top_k"
        }
        # Session identity in the api-key slot: the agent sets
        # api_key=_rollout["api_key"] on its LLM client, so every call to the
        # fixed gateway base_url carries "Authorization: Bearer <sid>".
        future = await client.invoke_async(
            payload=payload,
            session_id=session_id,
            input_id=session_id,
            base_url=server.base_url,
            api_key=session_id,
            model_id=cfg.model_id,
            sampling_params=agent_params,
        )

        result = await future.result_async(timeout=cfg.acr_timeout)
        episode_reward = _extract_reward(result)
        records = await gateway.finish_session(
            session_id,
            base_sample=BaseTrace(rollout_id=session_id, group_index=task_index),
            reward=episode_reward,
        )
        _log_records(session_id, records, episode_reward, task_index)

        if not records:
            noop = _make_noop_sample(group_index=task_index, session_id=session_id, status_name="FAILED")
            noop.metadata["episode_error"] = "no trace records captured"
            logger.info("Episode failed (session=%s): %s", session_id, noop.metadata["episode_error"])
            return [noop]

        samples = [
            _record_to_sample(rec, task_index, next(sample_counter), session_id, i) for i, rec in enumerate(records)
        ]
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
        # Idempotent: after a successful finish_session this is a no-op; on any
        # failure path it drains stragglers and discards the partial trajectory.
        await gateway.drop_session(session_id)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source,
    evaluation: bool = False,
):
    """Custom slime rollout function: ACR agents + in-process rollout gateway.

    Implements slime's --rollout-function-path interface.
    """
    _, RolloutFnTrainOutput, RolloutFnEvalOutput = _import_slime_types()
    server, client, cfg = _ensure_initialized(args)

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
    # (GRPO grouping in training, a unique id in eval); rows are flattened
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
                return await _process_one_episode(s, server, client, cfg, sampling_params, group_index, sample_counter)

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

    # ---- Step 3: per-batch summary ----
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
        # Episode reward is broadcast to every row-Sample; take the first.
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
