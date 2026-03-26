"""Shared utilities for evaluation scripts."""

import json
from pathlib import Path

import boto3
import tomllib


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


def prepare_payload(folder_uri: str, require_maximal_migration: bool = False, prompt_type: str = "baseline") -> dict:
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
        "require_maximal_migration": require_maximal_migration,
        "prompt_type": prompt_type,
    }


def append_result_to_file(result_path: Path, item_data: dict):
    """Append a single result to the JSON file (one JSON object per line)."""
    with open(result_path, "a") as f:
        f.write(json.dumps(item_data) + "\n")
