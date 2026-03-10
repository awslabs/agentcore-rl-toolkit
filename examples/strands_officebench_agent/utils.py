import json
import logging
import os
import shutil
import tarfile

import boto3

logger = logging.getLogger(__name__)

TESTBED_DIR = "/testbed"


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"{s3_uri} does not start with s3://")
    _, _, bucket_and_key = s3_uri.partition("s3://")
    bucket, _, key = bucket_and_key.partition("/")
    return bucket, key


def load_task_from_s3(task_uri: str) -> dict:
    """Download task config JSON from S3 and return as dict."""
    s3 = boto3.client("s3")
    bucket, key = parse_s3_uri(task_uri)
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    return json.loads(content)


def setup_testbed(testbed_uri: str | None) -> str:
    """Set up the testbed directory.

    If testbed_uri is provided, downloads and extracts tar.gz from S3 into /testbed/.
    Otherwise, creates empty /testbed/{data,calendar,emails} directories.

    Returns the testbed directory path.
    """
    # Clean up existing testbed contents (keep the directory itself,
    # which may be owned by root in Docker)
    if os.path.exists(TESTBED_DIR):
        for entry in os.listdir(TESTBED_DIR):
            entry_path = os.path.join(TESTBED_DIR, entry)
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)

    if testbed_uri:
        s3 = boto3.client("s3")
        bucket, key = parse_s3_uri(testbed_uri)

        # Check if testbed tar.gz exists (not all tasks have testbed data)
        from botocore.exceptions import ClientError

        try:
            s3.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info(f"No testbed archive at {testbed_uri}, using empty testbed")
                testbed_uri = None
            else:
                raise

    if testbed_uri:
        tar_path = "/tmp/testbed.tar.gz"
        s3.download_file(bucket, key, tar_path)

        # Extract tar.gz — the archive should contain the testbed contents
        # (data/, calendar/, emails/ directories)
        os.makedirs(TESTBED_DIR, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=TESTBED_DIR)

        os.remove(tar_path)
        logger.info(f"Extracted testbed from {testbed_uri} to {TESTBED_DIR}")
    else:
        logger.info("No testbed data, using empty testbed directories")

    # Ensure standard directories exist
    os.makedirs(os.path.join(TESTBED_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(TESTBED_DIR, "calendar"), exist_ok=True)
    os.makedirs(os.path.join(TESTBED_DIR, "emails"), exist_ok=True)

    logger.info(f"Testbed contents: {os.listdir(TESTBED_DIR)}")
    return TESTBED_DIR
