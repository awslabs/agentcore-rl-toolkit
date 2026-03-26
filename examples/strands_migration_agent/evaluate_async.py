"""Async evaluation script for migration agent using RolloutClient async APIs."""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from eval_utils import append_result_to_file, get_s3_folder_uris, load_config, prepare_payload

from agentcore_rl_toolkit import RolloutClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_batch_mode(client, payloads, s3_folder_uris, result_path, max_concurrent, timeout):
    """Run evaluation using client.run_batch_async() — managed async batch lifecycle."""
    completed = 0
    succeeded = 0
    failed = 0
    task_successes = 0

    async for item in client.run_batch_async(payloads, max_concurrent_sessions=max_concurrent, timeout=timeout):
        completed += 1

        record = {
            "index": item.index,
            "success": item.success,
            "input_uri": s3_folder_uris[item.index],
        }

        if item.success:
            succeeded += 1
            record["result"] = item.result
            record["elapsed"] = item.elapsed
            rewards = item.result.get("rewards")
            if rewards == 1:
                task_successes += 1
            logger.info(
                f"[{completed}/{len(payloads)}] Index {item.index} completed in {item.elapsed:.1f}s - "
                f"rewards: {rewards}"
            )
        else:
            failed += 1
            record["error"] = item.error
            record["elapsed"] = item.elapsed
            logger.warning(
                f"[{completed}/{len(payloads)}] Index {item.index} failed in {item.elapsed:.1f}s: {item.error}"
            )

        append_result_to_file(result_path, record)

    return succeeded, failed, task_successes


async def run_individual_mode(client, payloads, s3_folder_uris, result_path, timeout):
    """Run evaluation using invoke_async() + gather — two-step pattern for training frameworks."""
    # Step 1: Fire all invoke_async concurrently
    # (rate limiter paces at 25 TPS, cold starts don't block each other)
    # return_exceptions=True so one failed submission doesn't cancel the rest
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
            "input_uri": s3_folder_uris[idx],
            "success": False,
            "error": f"Submission failed: {exc}",
            "elapsed": 0.0,
        }
        logger.warning(f"[{failed}/{len(payloads)}] Index {idx} submission failed: {exc}")
        append_result_to_file(result_path, record)

    # Record gather results
    for i, result in enumerate(results):
        idx, future = futures[i]
        record = {
            "index": idx,
            "input_uri": s3_folder_uris[idx],
        }

        if isinstance(result, BaseException):
            failed += 1
            record["success"] = False
            record["error"] = str(result)
            record["elapsed"] = future.elapsed()
            logger.warning(
                f"[{succeeded + failed}/{len(payloads)}] Index {idx} failed " f"in {future.elapsed():.1f}s: {result}"
            )
        else:
            succeeded += 1
            record["success"] = True
            record["result"] = result
            record["elapsed"] = future.elapsed()
            rewards = result.get("rewards")
            if rewards == 1:
                task_successes += 1
            logger.info(
                f"[{succeeded + failed}/{len(payloads)}] Index {idx} completed "
                f"in {future.elapsed():.1f}s - rewards: {rewards}"
            )

        append_result_to_file(result_path, record)

    return succeeded, failed, task_successes


async def main():
    config = load_config()
    agentcore_config = config.get("agentcore", {})
    eval_config = config.get("eval", {})

    parser = argparse.ArgumentParser(description="Async evaluation of migration agent on benchmark")
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
        "--s3_input_bucket",
        type=str,
        default=eval_config.get("s3_input_bucket"),
        help="S3 bucket for retrieving input repositories",
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
        help="Limit number of repositories to evaluate (for testing)",
    )
    parser.add_argument(
        "--max_pool_connections",
        type=int,
        default=10,
        help="Max urllib3 connection pool size for boto3 clients (default: 10). "
        "If this value is smaller than --max_concurrent, you may see urllib3 warnings "
        "'Connection pool is full, discarding connection'. This is not an error — "
        "requests still succeed, but excess connections are created and discarded "
        "instead of being reused from the pool, adding minor TCP/TLS overhead. ",
    )
    parser.add_argument(
        "--sampling_params",
        type=str,
        default=eval_config.get("sampling_params"),
        help="Sampling parameters as JSON string (e.g. '{\"temperature\": 0.7}')",
    )
    parser.add_argument(
        "--require_maximal_migration",
        action="store_true",
        default=False,
        help="Whether a repository is evaluated under maximal migration",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="baseline",
        choices=["baseline", "rag", "hybrid"],
        help="Specify Java migration prompt type",
    )

    args = parser.parse_args()

    # Validation
    if not args.agent_arn:
        parser.error("--agent_arn is required (or set agentcore.agent_arn in config.toml)")
    if not args.s3_input_bucket:
        parser.error("--s3_input_bucket is required")
    if not args.s3_output_bucket:
        parser.error("--s3_output_bucket is required")

    # Get repository folders
    logger.info(f"Listing repositories from {args.s3_input_bucket}...")
    s3_folder_uris = get_s3_folder_uris(args.s3_input_bucket)
    if not s3_folder_uris:
        logger.error(f"No folders found in {args.s3_input_bucket}")
        return

    # Apply limit if specified
    if args.limit:
        s3_folder_uris = s3_folder_uris[: args.limit]

    logger.info(f"Found {len(s3_folder_uris)} repositories to evaluate")

    # Prepare payloads
    payloads = [prepare_payload(uri, args.require_maximal_migration, args.prompt_type) for uri in s3_folder_uris]

    # Setup results directory and file
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    result_path = results_dir / f"{args.exp_id}.jsonl"

    # Error if file already exists to prevent accidental overwrites
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
            client, payloads, s3_folder_uris, result_path, args.max_concurrent, args.timeout
        )
    else:
        logger.info(f"Individual mode: submitting all {len(payloads)} requests concurrently")
        succeeded, failed, task_successes = await run_individual_mode(
            client, payloads, s3_folder_uris, result_path, args.timeout
        )

    # Summary
    total_repos = len(payloads)
    success_rate = task_successes / total_repos if total_repos > 0 else 0
    total_time = time.time() - benchmark_start
    logger.info("=" * 50)
    logger.info(f"Evaluation complete: {succeeded} succeeded, {failed} failed")
    logger.info(f"Task success rate: {task_successes}/{total_repos} ({success_rate:.1%})")
    logger.info(f"Total benchmark time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    logger.info(f"Results saved to: {result_path}")


if __name__ == "__main__":
    asyncio.run(main())
