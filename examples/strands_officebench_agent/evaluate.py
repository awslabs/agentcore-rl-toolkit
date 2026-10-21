"""Batch evaluation script for OfficeBench agent through background HTTP invocations.

Each OfficeBench task directory can have multiple subtasks (0.json, 1.json, ...),
all sharing the same testbed. Each subtask is an independent test case.
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

import boto3

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


def list_all_subtasks(s3_uri: str) -> list[dict]:
    """List all subtask entries under the S3 prefix.

    Discovers the structure: {prefix}/{task_id}/{subtask_id}/config.json
    where testbed is at {prefix}/{task_id}/testbed.tar.gz.

    Returns sorted list of dicts with keys: task_id, subtask_id, task_uri, testbed_uri.
    """
    path = s3_uri.replace("s3://", "").rstrip("/")
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client("s3")
    entries = []
    paginator = s3.get_paginator("list_objects_v2")

    # Find all config.json files
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/config.json"):
                # Parse: {prefix}/{task_id}/{subtask_id}/config.json
                rel = key[len(prefix) :]
                parts = rel.split("/")
                if len(parts) == 3:  # task_id/subtask_id/config.json
                    task_id, subtask_id, _ = parts
                    entries.append(
                        {
                            "task_id": task_id,
                            "subtask_id": subtask_id,
                            "task_uri": f"s3://{bucket_name}/{key}",
                            "testbed_uri": f"s3://{bucket_name}/{prefix}{task_id}/testbed.tar.gz",
                        }
                    )

    # Sort by (category, task_num, subtask_id)
    def sort_key(e):
        parts = e["task_id"].split("-")
        try:
            return (int(parts[0]), int(parts[1]), int(e["subtask_id"]))
        except (ValueError, IndexError):
            return (999, 999, 999)

    entries.sort(key=sort_key)
    return entries


def prepare_payload(entry: dict) -> dict:
    """Prepare a single payload for a subtask entry."""
    return {
        "task_uri": entry["task_uri"],
        "testbed_uri": entry["testbed_uri"],
    }


def get_task_category(task_id: str) -> str:
    """Extract task category (1, 2, 3) from task ID like '1-1'."""
    return task_id.split("-")[0]


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
                "api_key": sid,
                "sampling_params": sampling or {},
            }
            if args.base_url:
                task["_config"]["base_url"] = args.base_url
            if args.model_id:
                task["_config"]["model_id"] = args.model_id
            record = {
                "index": index,
                "session_id": sid,
                "success": False,
                "task_id": entry["task_id"],
                "subtask_id": entry["subtask_id"],
                "category": get_task_category(entry["task_id"]),
                "task_uri": entry["task_uri"],
            }
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
    category_stats = {}
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run_one(i, entry)) for i, entry in enumerate(entries)]
            for future in asyncio.as_completed(tasks):
                record = await future
                completed += 1
                succeeded += int(record["success"])
                passed = record.get("reward") == 1.0
                task_successes += int(passed)
                if "category" in record:
                    record["is_pass"] = passed
                    stats = category_stats.setdefault(record["category"], {"total": 0, "passed": 0})
                    stats["total"] += 1
                    stats["passed"] += int(passed)
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
    for category, stats in sorted(category_stats.items()):
        logger.info(
            "%s-app tasks: %d/%d (%.1f%%)",
            category,
            stats["passed"],
            stats["total"],
            100 * stats["passed"] / stats["total"],
        )
    logger.info(
        "Overall: %d/%d (%.1f%%)",
        task_successes,
        len(entries),
        100 * task_successes / len(entries) if entries else 0.0,
    )
    logger.info("Total time=%.1fs results=%s", time.monotonic() - benchmark_start, result_path)


def main():
    config = load_config()
    agentcore_config = config.get("agentcore", {})
    eval_config = config.get("eval", {})

    parser = argparse.ArgumentParser(description="Evaluate OfficeBench agent on benchmark tasks")
    parser.add_argument(
        "--agent_arn",
        type=str,
        default=agentcore_config.get("agent_arn"),
        help="Agent ARN for ACR deployment",
    )
    parser.add_argument(
        "--s3_input_bucket",
        type=str,
        default=eval_config.get("s3_input_bucket"),
        help="S3 URI for task data (e.g. s3://bucket/officebench/)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=eval_config.get("base_url"),
        help="vLLM server URL for model inference (omit for Bedrock API)",
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
        default="eval",
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
        default=1800.0,
        help="Timeout in seconds per request (default: 1800s / 30 min)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of subtasks to evaluate (for testing)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["1", "2", "3"],
        help="Only evaluate tasks of a specific category (1-app, 2-app, 3-app)",
    )

    parser.add_argument("--tps_limit", type=int, default=5)
    parser.add_argument(
        "--sampling_params", default=eval_config.get("sampling_params"), help="Sampling parameters as JSON"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path(tempfile.gettempdir()) / "strands_officebench_agent-evaluations"
    )
    args = parser.parse_args()

    for field in ("max_concurrent", "tps_limit", "timeout"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field} must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    if not args.agent_arn:
        parser.error("--agent_arn is required (or set agentcore.agent_arn in config.toml)")
    if not args.s3_input_bucket:
        parser.error("--s3_input_bucket is required")
    if args.base_url and not args.model_id:
        parser.error("--model_id is required with --base_url")

    # List all subtasks
    logger.info(f"Listing subtasks from {args.s3_input_bucket}...")
    entries = list_all_subtasks(args.s3_input_bucket)
    if not entries:
        logger.error(f"No subtasks found in {args.s3_input_bucket}")
        return

    # Filter by category if specified
    if args.category:
        entries = [e for e in entries if get_task_category(e["task_id"]) == args.category]

    # Apply limit
    if args.limit:
        entries = entries[: args.limit]

    logger.info(f"Found {len(entries)} subtasks to evaluate")

    # Setup results directory
    results_dir = args.output_dir
    result_path = results_dir / f"{args.exp_id}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        logger.error(f"Results file already exists: {result_path}")
        logger.error("Delete the file or use a different --exp_id")
        return

    logger.info(f"Results will be written to: {result_path}")

    asyncio.run(evaluate(args, entries, result_path))


if __name__ == "__main__":
    main()
