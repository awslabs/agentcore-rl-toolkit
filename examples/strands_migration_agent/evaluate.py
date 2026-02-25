"""Evaluation script for migration agent using RolloutClient.run_batch()."""

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


def get_s3_folder_uris(s3_uri: str) -> list[str]:
    """
    Get full S3 URIs for folders under a specific S3 path.

    Args:
        s3_uri: Can be full URI (s3://bucket/prefix/) or just bucket name
    """
    path = s3_uri.replace("s3://", "")
    bucket_name = path.split("/")[0]
    prefix = "/".join(path.split("/")[1:])

    s3 = boto3.client("s3")

    folder_uris = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter="/"):
        for prefix_obj in page.get("CommonPrefixes", []):
            folder_path = prefix_obj["Prefix"]
            folder_uris.append(f"s3://{bucket_name}/{folder_path}")

    return folder_uris


def prepare_payload(folder_uri: str) -> dict:
    """
    Prepare a single payload for a repository folder.

    Args:
        folder_uri: S3 folder URI like "s3://migration-bench/15093015999__EJServer/"

    Returns:
        Payload dictionary (without _rollout config, RolloutClient adds that)
    """
    folder_uri = folder_uri.rstrip("/")
    repo_name = folder_uri.split("/")[-1]

    repo_uri = f"{folder_uri}/{repo_name}.tar.gz"
    metadata_uri = f"{folder_uri}/metadata.json"

    return {
        "prompt": "Please help migrate this repo: {repo_path}. There are {num_tests} test cases in it.",
        "repo_uri": repo_uri,
        "metadata_uri": metadata_uri,
        "require_maximal_migration": False,
    }


def append_result_to_file(result_path: Path, item_data: dict):
    """Append a single result to the JSON file (one JSON object per line)."""
    with open(result_path, "a") as f:
        f.write(json.dumps(item_data) + "\n")


def main():
    config = load_config()
    agentcore_config = config.get("agentcore", {})
    eval_config = config.get("eval", {})

    parser = argparse.ArgumentParser(description="Evaluate migration agent on benchmark")
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
        help="Limit number of repositories to evaluate (for testing)",
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
    payloads = [prepare_payload(uri) for uri in s3_folder_uris]

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
    task_successes = 0  # Tasks with reward = 1

    for item in client.run_batch(payloads, max_concurrent_sessions=args.max_concurrent, timeout=args.timeout):
        completed += 1

        # Build record with metadata
        record = {
            "index": item.index,
            "success": item.success,
            "input_uri": s3_folder_uris[item.index],
        }

        if item.success:
            succeeded += 1
            record["result"] = item.result
            record["elapsed"] = item.elapsed
            rewards = item.result.get("rewards", [])
            # Check if task succeeded (reward = 1)
            if rewards and rewards[0] == 1:
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

        # Append to file immediately (streaming)
        append_result_to_file(result_path, record)

    # Summary
    total_repos = len(payloads)
    success_rate = task_successes / total_repos if total_repos > 0 else 0
    total_time = time.time() - benchmark_start
    logger.info("=" * 50)
    logger.info(f"Evaluation complete: {succeeded} succeeded, {failed} failed")
    logger.info(f"Task success rate: {task_successes}/{total_repos} ({success_rate:.1%})")
    logger.info(f"Total benchmark time: {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
