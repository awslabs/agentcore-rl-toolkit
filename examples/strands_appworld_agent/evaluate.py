"""Async evaluation script for AppWorld agent using RolloutClient async APIs."""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import tomllib
from appworld.task import load_task_ids

from agentcore_rl_toolkit import RolloutClient

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


def append_result_to_file(result_path: Path, item_data: dict):
    """Append a single result to the JSONL file (one JSON object per line)."""
    with open(result_path, "a") as f:
        f.write(json.dumps(item_data) + "\n")


async def run_batch_mode(client, payloads, task_ids, result_path, max_concurrent, timeout):
    """Run evaluation using client.run_batch_async() -- managed async batch lifecycle."""
    completed = 0
    succeeded = 0
    failed = 0
    task_successes = 0

    async for item in client.run_batch_async(payloads, max_concurrent_sessions=max_concurrent, timeout=timeout):
        completed += 1

        record = {
            "index": item.index,
            "success": item.success,
            "task_id": task_ids[item.index],
        }

        if item.success:
            succeeded += 1
            record["result"] = item.result
            record["elapsed"] = item.elapsed
            rewards = item.result.get("rewards", 0.0)
            if rewards == 1.0:
                task_successes += 1
            logger.info(
                f"[{completed}/{len(payloads)}] {task_ids[item.index]} completed in {item.elapsed:.1f}s - "
                f"reward: {rewards}"
            )
        else:
            failed += 1
            record["error"] = item.error
            record["elapsed"] = item.elapsed
            logger.warning(
                f"[{completed}/{len(payloads)}] {task_ids[item.index]} failed in {item.elapsed:.1f}s: {item.error}"
            )

        append_result_to_file(result_path, record)

    return succeeded, failed, task_successes


async def run_individual_mode(client, payloads, task_ids, result_path, timeout):
    """Run evaluation using invoke_async() + gather -- fire all, then collect all results."""
    # Step 1: Fire all invoke_async concurrently
    submit_tasks = [asyncio.create_task(client.invoke_async(p)) for p in payloads]
    submit_results = await asyncio.gather(*submit_tasks, return_exceptions=True)

    # Separate successful futures from submission failures
    futures = []  # (idx, RolloutFuture)
    submit_failures = []  # (idx, Exception)
    for idx, result in enumerate(submit_results):
        if isinstance(result, BaseException):
            submit_failures.append((idx, result))
        else:
            futures.append((idx, result))

    if submit_failures:
        logger.warning(f"{len(submit_failures)} submissions failed, {len(futures)} succeeded")
    logger.info(f"{len(futures)} requests submitted, gathering results...")

    # Step 2: Gather all results concurrently
    results = await asyncio.gather(
        *[f.result_async(timeout=timeout) for _, f in futures],
        return_exceptions=True,
    )

    # Process results
    succeeded = 0
    failed = 0
    task_successes = 0

    # Record submission failures first
    for idx, exc in submit_failures:
        failed += 1
        record = {
            "index": idx,
            "task_id": task_ids[idx],
            "success": False,
            "error": f"Submission failed: {exc}",
            "elapsed": 0.0,
        }
        logger.warning(f"[{failed}/{len(payloads)}] {task_ids[idx]} submission failed: {exc}")
        append_result_to_file(result_path, record)

    # Record gather results
    for i, result in enumerate(results):
        idx, future = futures[i]
        record = {
            "index": idx,
            "task_id": task_ids[idx],
        }

        if isinstance(result, BaseException):
            failed += 1
            record["success"] = False
            record["error"] = str(result)
            record["elapsed"] = future.elapsed()
            logger.warning(
                f"[{succeeded + failed}/{len(payloads)}] {task_ids[idx]} failed "
                f"in {future.elapsed():.1f}s: {result}"
            )
        else:
            succeeded += 1
            record["success"] = True
            record["result"] = result
            record["elapsed"] = future.elapsed()
            rewards = result.get("rewards", 0.0)
            if rewards == 1.0:
                task_successes += 1
            logger.info(
                f"[{succeeded + failed}/{len(payloads)}] {task_ids[idx]} completed "
                f"in {future.elapsed():.1f}s - reward: {rewards}"
            )

        append_result_to_file(result_path, record)

    return succeeded, failed, task_successes


async def main():
    config = load_config()
    agentcore_config = config.get("agentcore", {})
    eval_config = config.get("eval", {})

    parser = argparse.ArgumentParser(description="Async evaluation of AppWorld agent on benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "individual"],
        default="batch",
        help="Evaluation mode: 'batch' uses run_batch_async, 'individual' uses invoke_async + gather",
    )
    parser.add_argument(
        "--agent_arn",
        type=str,
        default=agentcore_config.get("agent_arn"),
        help="Agent ARN (example: arn:aws:bedrock-agentcore:{region}:{account_id}:runtime/{agent_id})",
    )
    parser.add_argument(
        "--s3_output_bucket",
        type=str,
        default=eval_config.get("s3_output_bucket"),
        help="S3 bucket for storing rollout results",
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
        help="Max concurrent ACR sessions (batch mode only)",
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
        "--max_pool_connections",
        type=int,
        default=10,
        help="Max urllib3 connection pool size for boto3 clients (default: 10). "
        "If this value is smaller than --max_concurrent, you may see urllib3 warnings "
        "'Connection pool is full, discarding connection'. This is not an error -- "
        "requests still succeed, but excess connections are created and discarded "
        "instead of being reused from the pool, adding minor TCP/TLS overhead.",
    )
    parser.add_argument(
        "--sampling_params",
        type=str,
        default=eval_config.get("sampling_params"),
        help="Sampling parameters as JSON string (e.g. '{\"temperature\": 0.7}')",
    )

    args = parser.parse_args()

    # Validation
    if not args.agent_arn:
        parser.error("--agent_arn is required (or set agentcore.agent_arn in config.toml)")
    if not args.s3_output_bucket:
        parser.error("--s3_output_bucket is required")

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

    # Prepare payloads
    payloads = [prepare_payload(tid) for tid in task_ids]

    # Setup results directory and file
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    result_path = results_dir / f"{args.exp_id}.jsonl"

    if result_path.exists():
        logger.error(f"Results file already exists: {result_path}")
        logger.error("Delete the file or use a different --exp_id")
        return

    logger.info(f"Results will be written to: {result_path}")

    # Parse sampling params
    sampling_params = {}
    if args.sampling_params:
        if isinstance(args.sampling_params, str):
            sampling_params = json.loads(args.sampling_params)
        else:
            sampling_params = dict(args.sampling_params)

    # Create client
    client = RolloutClient(
        agent_runtime_arn=args.agent_arn,
        s3_bucket=args.s3_output_bucket,
        exp_id=args.exp_id,
        base_url=args.base_url,
        model_id=args.model_id,
        max_pool_connections=args.max_pool_connections,
        sampling_params=sampling_params,
    )

    # Run evaluation
    logger.info(f"Starting async evaluation (mode={args.mode}, timeout={args.timeout}s)...")
    benchmark_start = time.time()

    if args.mode == "batch":
        logger.info(f"Batch mode: max_concurrent={args.max_concurrent}")
        succeeded, failed, task_successes = await run_batch_mode(
            client, payloads, task_ids, result_path, args.max_concurrent, args.timeout
        )
    else:
        logger.info(f"Individual mode: submitting all {len(payloads)} requests concurrently")
        succeeded, failed, task_successes = await run_individual_mode(
            client, payloads, task_ids, result_path, args.timeout
        )

    # Summary
    total_tasks = len(payloads)
    success_rate = task_successes / total_tasks if total_tasks > 0 else 0
    total_time = time.time() - benchmark_start
    logger.info("=" * 50)
    logger.info(f"Evaluation complete: {succeeded} succeeded, {failed} failed")
    logger.info(f"Task success rate: {task_successes}/{total_tasks} ({success_rate:.1%})")
    logger.info(f"Total benchmark time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    logger.info(f"Results saved to: {result_path}")


if __name__ == "__main__":
    asyncio.run(main())
