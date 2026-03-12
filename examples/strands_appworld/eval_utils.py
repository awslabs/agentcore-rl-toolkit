"""Shared utilities for AppWorld evaluation scripts."""

import json
import logging
from pathlib import Path

import boto3
import tomllib

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration from TOML file if it exists."""
    path = Path(__file__).parent / config_path
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}


def load_task_ids(path: str) -> list[str]:
    """Load task IDs from a txt file (one per line).

    Args:
        path: Local file path or S3 URI (s3://bucket/key).

    Returns:
        List of task ID strings.
    """
    if path.startswith("s3://"):
        return _load_task_ids_from_s3(path)
    return _load_task_ids_from_local(path)


def _load_task_ids_from_local(path: str) -> list[str]:
    """Load task IDs from a local txt file."""
    with open(path) as f:
        lines = f.read().strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def _load_task_ids_from_s3(s3_uri: str) -> list[str]:
    """Load task IDs from an S3 txt file."""
    s3_path = s3_uri.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    key = "/".join(s3_path.split("/")[1:])

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8").strip()
    return [line.strip() for line in content.splitlines() if line.strip()]


def prepare_payload(task_id: str) -> dict:
    """Prepare a single payload for an AppWorld task.

    Args:
        task_id: AppWorld task ID (e.g. "ff58e36_2")

    Returns:
        Payload dictionary (without _rollout config, RolloutClient adds that)
    """
    return {"task_id": task_id}


def append_result_to_file(result_path: Path, item_data: dict):
    """Append a single result to the JSONL file (one JSON object per line)."""
    with open(result_path, "a") as f:
        f.write(json.dumps(item_data) + "\n")
