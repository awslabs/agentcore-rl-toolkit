"""Batch evaluation script for OfficeBench agent using RolloutClient.run_batch().

Each OfficeBench task directory can have multiple subtasks (0.json, 1.json, ...),
all sharing the same testbed. Each subtask is an independent test case.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import boto3
import tomllib

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
                rel = key[len(prefix):]
                parts = rel.split("/")
                if len(parts) == 3:  # task_id/subtask_id/config.json
                    task_id, subtask_id, _ = parts
                    entries.append({
                        "task_id": task_id,
                        "subtask_id": subtask_id,
                        "task_uri": f"s3://{bucket_name}/{key}",
                        "testbed_uri": f"s3://{bucket_name}/{prefix}{task_id}/testbed.tar.gz",
                    })

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


def append_result_to_file(result_path: Path, item_data: dict):
    """Append a single result to the JSONL file."""
    with open(result_path, "a") as f:
        f.write(json.dumps(item_data) + "\n")


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
        "--s3_output_bucket",
        type=str,
        default=eval_config.get("s3_output_bucket"),
        help="S3 bucket for storing rollout results",
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

    args = parser.parse_args()

    if not args.agent_arn:
        parser.error("--agent_arn is required (or set agentcore.agent_arn in config.toml)")
    if not args.s3_input_bucket:
        parser.error("--s3_input_bucket is required")
    if not args.s3_output_bucket:
        parser.error("--s3_output_bucket is required")

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

    # Prepare payloads
    payloads = [prepare_payload(e) for e in entries]

    # Setup results directory
    results_dir = Path(__file__).parent / "results"
    result_path = results_dir / f"{args.exp_id}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        logger.error(f"Results file already exists: {result_path}")
        logger.error("Delete the file or use a different --exp_id")
        return

    logger.info(f"Results will be written to: {result_path}")

    # Create client
    client = RolloutClient(
        agent_runtime_arn=args.agent_arn,
        s3_bucket=args.s3_output_bucket,
        exp_id=args.exp_id,
        base_url=args.base_url,
        model_id=args.model_id,
    )

    # Run batch and stream results
    logger.info(f"Starting evaluation with max_concurrent={args.max_concurrent}, timeout={args.timeout}s...")

    benchmark_start = time.time()
    completed = 0
    succeeded = 0
    failed = 0
    # Track pass/fail by category
    category_stats: dict[str, dict[str, int]] = {}

    for item in client.run_batch(payloads, max_concurrent_sessions=args.max_concurrent, timeout=args.timeout):
        completed += 1
        entry = entries[item.index]
        task_id = entry["task_id"]
        subtask_id = entry["subtask_id"]
        category = get_task_category(task_id)
        display_id = f"{task_id}/{subtask_id}"

        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}
        category_stats[category]["total"] += 1

        record = {
            "index": item.index,
            "task_id": task_id,
            "subtask_id": subtask_id,
            "category": category,
            "success": item.success,
            "task_uri": entry["task_uri"],
        }

        if item.success:
            succeeded += 1
            record["result"] = item.result
            record["elapsed"] = item.elapsed
            rewards = item.result.get("rewards", 0.0)
            is_pass = rewards == 1.0
            record["is_pass"] = is_pass
            if is_pass:
                category_stats[category]["passed"] += 1
            logger.info(
                f"[{completed}/{len(payloads)}] {display_id} completed in {item.elapsed:.1f}s - "
                f"reward: {rewards} ({'PASS' if is_pass else 'FAIL'})"
            )
        else:
            failed += 1
            record["error"] = item.error
            record["elapsed"] = item.elapsed
            logger.warning(
                f"[{completed}/{len(payloads)}] {display_id} failed in {item.elapsed:.1f}s: {item.error}"
            )

        append_result_to_file(result_path, record)

    # Summary
    total_time = time.time() - benchmark_start
    logger.info("=" * 60)
    logger.info(f"Evaluation complete: {succeeded} succeeded, {failed} failed")
    for cat in sorted(category_stats):
        stats = category_stats[cat]
        rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        logger.info(f"  {cat}-app tasks: {stats['passed']}/{stats['total']} ({rate:.1%})")
    total_tasks = len(payloads)
    total_passed = sum(s["passed"] for s in category_stats.values())
    overall_rate = total_passed / total_tasks if total_tasks > 0 else 0
    logger.info(f"  Overall: {total_passed}/{total_tasks} ({overall_rate:.1%})")
    logger.info(f"Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    logger.info(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
