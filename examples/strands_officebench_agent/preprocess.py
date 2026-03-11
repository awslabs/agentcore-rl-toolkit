"""Package OfficeBench tasks for S3 consumption.

Uploads task configs and testbed data to S3 in a format
that dev_app.py can consume during ACR invocations.

Each task directory can contain multiple subtasks (0.json, 1.json, ...),
and all subtasks share the same testbed. Each subtask is uploaded as a
separate entry keyed by "{task_id}/{subtask_id}".

S3 structure:
    s3://bucket/officebench/
    ├── 1-1/
    │   ├── 0/config.json        # subtask 0 config
    │   ├── 1/config.json        # subtask 1 config
    │   ├── ...
    │   └── testbed.tar.gz       # shared testbed data (if any)
    ├── 1-2/
    │   ├── 0/config.json
    │   └── testbed.tar.gz
    └── ...
"""

import argparse
import json
import logging
import os
import tarfile
import tempfile

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tar_directory(source_dir: str, output_path: str):
    """Create a tar.gz archive of a directory's contents."""
    with tarfile.open(output_path, "w:gz") as tar:
        for entry in os.listdir(source_dir):
            tar.add(os.path.join(source_dir, entry), arcname=entry)


def upload_task(
    s3_client,
    task_dir: str,
    task_id: str,
    s3_bucket: str,
    s3_prefix: str,
) -> int:
    """Upload all subtasks for a single task to S3.

    Returns the number of subtasks uploaded.
    """
    subtasks_dir = os.path.join(task_dir, "subtasks")
    if not os.path.exists(subtasks_dir):
        logger.warning(f"Skipping {task_id}: no subtasks/ directory")
        return 0

    subtask_files = sorted(f for f in os.listdir(subtasks_dir) if f.endswith(".json"))
    if not subtask_files:
        logger.warning(f"Skipping {task_id}: no subtask JSON files")
        return 0

    # Upload each subtask config
    for subtask_file in subtask_files:
        subtask_id = subtask_file.removesuffix(".json")
        config_path = os.path.join(subtasks_dir, subtask_file)
        s3_key = f"{s3_prefix}/{task_id}/{subtask_id}/config.json"
        s3_client.upload_file(config_path, s3_bucket, s3_key)
        logger.debug(f"Uploaded {config_path} -> s3://{s3_bucket}/{s3_key}")

    # Upload testbed once per task (shared across subtasks).
    # Also includes reference/ files and a cache/ snapshot of original data
    # so that evaluation paths resolve correctly.
    testbed_dir = os.path.join(task_dir, "testbed")
    reference_dir = os.path.join(task_dir, "reference")
    has_testbed = os.path.exists(testbed_dir) and os.listdir(testbed_dir)
    has_reference = os.path.exists(reference_dir) and os.listdir(reference_dir)

    if has_testbed or has_reference:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=True) as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                # Add testbed contents at root (data/, calendar/, emails/)
                if has_testbed:
                    for entry in os.listdir(testbed_dir):
                        tar.add(os.path.join(testbed_dir, entry), arcname=entry)

                # Add reference/ directory for evaluate_exact_match
                if has_reference:
                    for entry in os.listdir(reference_dir):
                        tar.add(
                            os.path.join(reference_dir, entry),
                            arcname=f"reference/{entry}",
                        )

                # Add cache/ snapshot of original testbed data for evaluate_diff_contain_text.
                # The cache stores the original data files before agent modification.
                if has_testbed:
                    data_dir = os.path.join(testbed_dir, "data")
                    if os.path.exists(data_dir):
                        for entry in os.listdir(data_dir):
                            tar.add(
                                os.path.join(data_dir, entry),
                                arcname=f"cache/data/{entry}",
                            )

            s3_key = f"{s3_prefix}/{task_id}/testbed.tar.gz"
            s3_client.upload_file(tmp.name, s3_bucket, s3_key)
            logger.debug(f"Uploaded testbed -> s3://{s3_bucket}/{s3_key}")

    return len(subtask_files)


def main():
    parser = argparse.ArgumentParser(description="Package OfficeBench tasks for S3")
    parser.add_argument(
        "--officebench_dir",
        type=str,
        required=True,
        help="Path to OfficeBench repository (e.g. /path/to/OfficeBench)",
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        required=True,
        help="S3 bucket name",
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="officebench",
        help="S3 key prefix (default: officebench)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["1", "2", "3"],
        help="Only upload tasks of a specific category",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks (directories) to upload",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list tasks, don't upload",
    )

    args = parser.parse_args()

    tasks_dir = os.path.join(args.officebench_dir, "tasks")
    if not os.path.exists(tasks_dir):
        logger.error(f"Tasks directory not found: {tasks_dir}")
        return

    # Collect task IDs
    task_ids = sorted(os.listdir(tasks_dir))

    # Filter by category
    if args.category:
        task_ids = [t for t in task_ids if t.startswith(f"{args.category}-")]

    # Sort by numeric ID
    def sort_key(task_id):
        parts = task_id.split("-")
        try:
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return (999, 999)

    task_ids.sort(key=sort_key)

    if args.limit:
        task_ids = task_ids[: args.limit]

    # Count total subtasks
    all_entries = []  # (task_id, subtask_id) pairs
    for task_id in task_ids:
        subtasks_dir = os.path.join(tasks_dir, task_id, "subtasks")
        if os.path.exists(subtasks_dir):
            for f in sorted(os.listdir(subtasks_dir)):
                if f.endswith(".json"):
                    all_entries.append((task_id, f.removesuffix(".json")))

    logger.info(f"Found {len(task_ids)} task dirs with {len(all_entries)} total subtasks")

    if args.dry_run:
        for task_id in task_ids:
            task_dir = os.path.join(tasks_dir, task_id)
            subtasks_dir = os.path.join(task_dir, "subtasks")
            num_subtasks = (
                len([f for f in os.listdir(subtasks_dir) if f.endswith(".json")]) if os.path.exists(subtasks_dir) else 0
            )
            has_testbed = os.path.exists(os.path.join(task_dir, "testbed"))
            logger.info(f"  {task_id}: {num_subtasks} subtasks (testbed: {has_testbed})")
        return

    s3 = boto3.client("s3")
    total_subtasks = 0
    for i, task_id in enumerate(task_ids):
        task_dir = os.path.join(tasks_dir, task_id)
        count = upload_task(s3, task_dir, task_id, args.s3_bucket, args.s3_prefix)
        total_subtasks += count
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(task_ids)} task dirs ({total_subtasks} subtasks)")

    logger.info(
        f"Done! Uploaded {total_subtasks} subtasks from {len(task_ids)} tasks to s3://{args.s3_bucket}/{args.s3_prefix}/"
    )

    # Write manifest with all (task_id, subtask_id) entries
    manifest = {
        "entries": [{"task_id": t, "subtask_id": s} for t, s in all_entries],
        "s3_prefix": f"s3://{args.s3_bucket}/{args.s3_prefix}/",
    }
    manifest_key = f"{args.s3_prefix}/manifest.json"
    s3.put_object(
        Bucket=args.s3_bucket,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=2),
    )
    logger.info(f"Manifest written to s3://{args.s3_bucket}/{manifest_key}")


if __name__ == "__main__":
    main()
