"""Evaluation script for AppWorld agent using RolloutClient.run_batch()."""

import argparse
import json
import logging
import time
from pathlib import Path

from eval_utils import append_result_to_file, load_config, load_task_ids, prepare_payload

from agentcore_rl_toolkit import RolloutClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    agentcore_config = config.get("agentcore", {})
    eval_config = config.get("eval", {})

    parser = argparse.ArgumentParser(description="Evaluate AppWorld agent on benchmark")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to task ID file (local path or s3://bucket/key). One task ID per line.",
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

    args = parser.parse_args()

    # Validation
    if not args.agent_arn:
        parser.error("--agent_arn is required (or set agentcore.agent_arn in config.toml)")
    if not args.s3_output_bucket:
        parser.error("--s3_output_bucket is required")

    # Load task IDs
    logger.info(f"Loading task IDs from {args.input}...")
    task_ids = load_task_ids(args.input)
    if not task_ids:
        logger.error(f"No task IDs found in {args.input}")
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

    # Run batch and stream results
    logger.info(f"Starting evaluation with max_concurrent={args.max_concurrent}, timeout={args.timeout}s...")

    benchmark_start = time.time()
    completed = 0
    succeeded = 0
    failed = 0
    reward_sum = 0.0

    for item in client.run_batch(payloads, max_concurrent_sessions=args.max_concurrent, timeout=args.timeout):
        completed += 1

        # Build record with metadata
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
            reward_val = rewards if isinstance(rewards, (int, float)) else (rewards[0] if rewards else 0.0)
            reward_sum += reward_val
            logger.info(
                f"[{completed}/{len(payloads)}] Task {task_ids[item.index]} completed in {item.elapsed:.1f}s - "
                f"reward: {reward_val}"
            )
        else:
            failed += 1
            record["error"] = item.error
            record["elapsed"] = item.elapsed
            logger.warning(
                f"[{completed}/{len(payloads)}] Task {task_ids[item.index]} failed in {item.elapsed:.1f}s: "
                f"{item.error}"
            )

        # Append to file immediately (streaming)
        append_result_to_file(result_path, record)

    # Summary
    total_tasks = len(payloads)
    avg_reward = reward_sum / total_tasks if total_tasks > 0 else 0
    total_time = time.time() - benchmark_start
    logger.info("=" * 50)
    logger.info(f"Evaluation complete: {succeeded} succeeded, {failed} failed")
    logger.info(f"Average reward: {avg_reward:.4f} (sum: {reward_sum:.2f}/{total_tasks})")
    logger.info(f"Total benchmark time: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
