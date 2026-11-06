"""Evaluate the SealQA agent on AgentCore Runtime through background HTTP invocations.

One invocation per (task, repeat); records go to OUTPUT_DIR/EXP_ID.jsonl. The reported metric is the mean reward
(accuracy, the share graded A), avg@repeats.
"""

import argparse
import asyncio
import json
import logging
import math
import tempfile
import time
import tomllib
import uuid
from pathlib import Path

from tasks import load_split

from agentcore_rl_toolkit.aws_tools.agentcore_tools import close_shared_agentcore_clients
from agentcore_rl_toolkit.aws_tools.persistent_dict import NullPersister, PersistentDict
from agentcore_rl_toolkit.concurrency.rate_limiter import LocalRateLimiter
from agentcore_rl_toolkit.rollout_session.agentcore_http_session import AgentCoreHttpSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration from TOML file if it exists."""
    path = Path(__file__).parent / config_path
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


async def evaluate(args, jobs, result_path):
    benchmark_start = time.monotonic()
    semaphore = asyncio.Semaphore(args.max_concurrent)
    starts = LocalRateLimiter(args.tps_limit)
    model_config = {
        "base_url": args.base_url,
        "model_id": args.model_id,
        "bedrock_fields": args.bedrock_fields,
        "sampling_params": args.sampling_params or {},
    }

    async def run_one(task, repeat):
        async with semaphore:
            sid = str(uuid.uuid4())
            started = time.monotonic()
            record = {"task_id": task["task_id"], "repeat": repeat, "session_id": sid, "success": False}
            try:
                async with AgentCoreHttpSession(
                    sid, PersistentDict(persister=NullPersister()), agentcore_runtime_arn=args.agent_arn
                ) as session:
                    await starts.wait_async()
                    async with asyncio.timeout(args.timeout):
                        dump = await session.run({**task, "_config": {**model_config, "api_key": sid}})
                if dump.exception:
                    raise RuntimeError(dump.exception)
                result = dump.task_output or {}
                reward = result.get("reward")
                if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                    raise ValueError(f"Expected a finite scalar reward, got {reward!r}")
                record.update(success=True, result=result, reward=reward)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["elapsed"] = time.monotonic() - started
            return record

    rewards, refusals = [], 0
    try:
        async with asyncio.TaskGroup() as group:
            futures = [group.create_task(run_one(task, repeat)) for task, repeat in jobs]
            for n, future in enumerate(asyncio.as_completed(futures), 1):
                record = await future
                if record["success"]:
                    rewards.append(record["reward"])
                    refusals += record["result"].get("stop") == "refusal"
                with open(result_path, "a") as stream:
                    stream.write(json.dumps(record) + "\n")
                logger.info(
                    "[%d/%d] %s r%d elapsed=%.1fs reward=%s error=%s",
                    n,
                    len(jobs),
                    record["task_id"],
                    record["repeat"],
                    record["elapsed"],
                    record.get("reward"),
                    record.get("error"),
                )
    finally:
        await close_shared_agentcore_clients()
    logger.info("Successful invocations: %d/%d", len(rewards), len(jobs))
    logger.info("Accuracy: %.4f", sum(rewards) / len(rewards) if rewards else 0.0)
    logger.info("Refusals (scored 0): %d", refusals)
    logger.info("Total time=%.1fs results=%s", time.monotonic() - benchmark_start, result_path)


def main():
    config = load_config()
    agentcore_config = config.get("agentcore", {})
    eval_config = config.get("eval", {})

    parser = argparse.ArgumentParser(description="Async evaluation of the SealQA agent")
    parser.add_argument("--agent_arn", default=agentcore_config.get("agent_arn"), help="AgentCore Runtime ARN")
    parser.add_argument("--model_id", default=eval_config.get("model_id"), help="Model ID for inference")
    parser.add_argument(
        "--base_url",
        default=eval_config.get("base_url"),
        help="OpenAI-compatible endpoint (vLLM); without it, model_id is a Bedrock model",
    )
    parser.add_argument(
        "--bedrock_fields",
        type=json.loads,
        default=eval_config.get("bedrock_fields"),
        help='additionalModelRequestFields of a Bedrock model, e.g. \'{"reasoning": {"effort": "high"}}\'',
    )
    parser.add_argument(
        "--sampling_params",
        type=json.loads,
        default=eval_config.get("sampling_params"),
        help="Sampling parameters as JSON (e.g. '{\"temperature\": 0.7}')",
    )
    parser.add_argument("--split", default="all", help="all, pilot50 or smoke2")
    parser.add_argument("--repeats", type=int, default=1, help="invocations per task (avg@repeats)")
    parser.add_argument("--exp_id", default="eval_async", help="Experiment ID for organizing results")
    parser.add_argument("--max_concurrent", type=int, default=32, help="Max concurrent ACR sessions")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Timeout in seconds per request")
    parser.add_argument("--tps_limit", type=int, default=5)
    parser.add_argument(
        "--output_dir", type=Path, default=Path(tempfile.gettempdir()) / "strands_sealqa_agent-evaluations"
    )
    args = parser.parse_args()

    for field in ("max_concurrent", "tps_limit", "timeout", "repeats"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field} must be positive")
    if not args.model_id:
        parser.error("--model_id is required")
    if not args.agent_arn:
        parser.error("--agent_arn is required (or set agentcore.agent_arn in config.toml)")

    tasks = load_split(args.split)
    jobs = [(task, repeat) for task in tasks for repeat in range(args.repeats)]
    logger.info("Split %s: %d tasks x %d repeats", args.split, len(tasks), args.repeats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{args.exp_id}.jsonl"
    if result_path.exists():
        logger.error(f"Results file already exists: {result_path}")
        logger.error("Delete the file or use a different --exp_id")
        return
    logger.info(f"Results will be written to: {result_path}")

    asyncio.run(evaluate(args, jobs, result_path))


if __name__ == "__main__":
    main()
