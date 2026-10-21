"""Evaluate AppWorld through background HTTP invocations."""

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

from appworld.task import load_task_ids

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


def prepare_payload(task_id: str) -> dict:
    """Payload is just task_id -- all data loaded server-side from baked-in AppWorld data."""
    return {"task_id": task_id}


async def evaluate(args, entries, result_path):
    benchmark_start = time.monotonic()
    semaphore = asyncio.Semaphore(args.max_concurrent)
    starts = LocalRateLimiter(args.tps_limit)
    sampling = args.sampling_params
    if isinstance(sampling, str):
        sampling = json.loads(sampling)

    async def run_one(index, entry):
        async with semaphore:
            sid = str(uuid.uuid4())
            started = time.monotonic()
            task = prepare_payload(entry)
            task["_config"] = {
                "base_url": args.base_url,
                "model_id": args.model_id,
                "api_key": sid,
                "sampling_params": sampling or {},
            }
            record = {"index": index, "session_id": sid, "success": False, "task_id": entry}
            try:
                async with AgentCoreHttpSession(
                    sid, PersistentDict(persister=NullPersister()), agentcore_runtime_arn=args.agent_arn
                ) as session:
                    await starts.wait_async()
                    async with asyncio.timeout(args.timeout):
                        dump = await session.run(task)
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

    completed = succeeded = task_successes = 0
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run_one(i, entry)) for i, entry in enumerate(entries)]
            for future in asyncio.as_completed(tasks):
                record = await future
                completed += 1
                succeeded += int(record["success"])
                passed = record.get("reward") == 1.0
                task_successes += int(passed)
                with open(result_path, "a") as stream:
                    stream.write(json.dumps(record) + "\n")
                logger.info(
                    "[%d/%d] index=%d elapsed=%.1fs reward=%s error=%s",
                    completed,
                    len(entries),
                    record["index"],
                    record["elapsed"],
                    record.get("reward"),
                    record.get("error"),
                )
    finally:
        await close_shared_agentcore_clients()
    logger.info("Completed=%d successful_invocations=%d task_successes=%d", completed, succeeded, task_successes)
    logger.info(
        "Task success rate: %d/%d (%.1f%%)",
        task_successes,
        len(entries),
        100 * task_successes / len(entries) if entries else 0.0,
    )
    logger.info("Total time=%.1fs results=%s", time.monotonic() - benchmark_start, result_path)


def main():
    config = load_config()
    agentcore_config = config.get("agentcore", {})
    eval_config = config.get("eval", {})

    parser = argparse.ArgumentParser(description="Async evaluation of AppWorld agent on benchmark")
    parser.add_argument(
        "--agent_arn",
        type=str,
        default=agentcore_config.get("agent_arn"),
        help="Agent ARN (example: arn:aws:bedrock-agentcore:{region}:{account_id}:runtime/{agent_id})",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=eval_config.get("base_url"),
        help="vLLM server URL for model inference",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=eval_config.get("model_id"),
        help="Model ID for inference",
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default="eval_async",
        help="Experiment ID for organizing results",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=100,
        help="Max concurrent ACR sessions",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Timeout in seconds per request (default: 3600s / 60 min)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate (for testing)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "dev", "test_normal", "test_challenge"],
        help="AppWorld dataset split to evaluate",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Only evaluate tasks of a specific difficulty level",
    )
    parser.add_argument(
        "--sampling_params",
        type=str,
        default=eval_config.get("sampling_params"),
        help="Sampling parameters as JSON string (e.g. '{\"temperature\": 0.7}')",
    )

    parser.add_argument("--tps_limit", type=int, default=5)
    parser.add_argument(
        "--output_dir", type=Path, default=Path(tempfile.gettempdir()) / "strands_appworld_agent-evaluations"
    )
    args = parser.parse_args()

    for field in ("max_concurrent", "tps_limit", "timeout"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    for field in ("base_url", "model_id"):
        if not getattr(args, field):
            parser.error(f"--{field} is required")

    # Validation
    if not args.agent_arn:
        parser.error("--agent_arn is required (or set agentcore.agent_arn in config.toml)")

    # Get task IDs from AppWorld
    logger.info(f"Loading task IDs for dataset={args.dataset}, difficulty={args.difficulty}...")
    load_kwargs = {"dataset_name": args.dataset}
    if args.difficulty is not None:
        load_kwargs["difficulty"] = args.difficulty
    task_ids = load_task_ids(**load_kwargs)

    if not task_ids:
        logger.error(f"No tasks found for dataset={args.dataset}, difficulty={args.difficulty}")
        return

    # Apply limit if specified
    if args.limit:
        task_ids = task_ids[: args.limit]

    logger.info(f"Found {len(task_ids)} tasks to evaluate")

    # Setup results directory and file
    results_dir = args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{args.exp_id}.jsonl"

    if result_path.exists():
        logger.error(f"Results file already exists: {result_path}")
        logger.error("Delete the file or use a different --exp_id")
        return

    logger.info(f"Results will be written to: {result_path}")

    asyncio.run(evaluate(args, task_ids, result_path))


if __name__ == "__main__":
    main()
