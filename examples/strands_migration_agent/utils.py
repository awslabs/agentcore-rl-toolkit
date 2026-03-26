import json
import logging
import os
import shutil
import subprocess
import tarfile

import boto3
from java_migration_agent.preprocessing import update_dependency_version, update_jdk_related

logger = logging.getLogger(__name__)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"{s3_uri} does not start with s3://")

    _, _, bucket_and_key = s3_uri.partition("s3://")
    bucket, _, key = bucket_and_key.partition("/")

    return bucket, key


def load_metadata_from_s3(s3_uri: str) -> dict:
    s3 = boto3.client("s3")
    bucket, key = parse_s3_uri(s3_uri)

    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    return json.loads(content)


def setup_repo_environment(repo_path: str, prompt_type: str = "baseline"):
    """
    1. Pre-warm Maven caches (best-effort)
    2. Make sure git works.
    """
    # Run full build lifecycle to pre-warm all caches: project dependencies,
    # plugin dependencies, and plugin runtime downloads (e.g. frontend-maven-plugin
    # fetching Node.js) that `dependency:go-offline` would miss.
    # The build will likely fail (repo is still Java 8), but that's fine —
    # the goal is to cache downloads so the agent's later `mvn clean verify` is quieter.
    result = subprocess.run(
        ["mvn", "clean", "verify"],
        cwd=repo_path,
        capture_output=True,
    )
    if result.returncode == 0:
        logger.info("Pre-warm build succeeded")
    else:
        logger.info("Pre-warm build failed (expected — repo not yet migrated)")

    # ensure git works
    subprocess.run(
        ["git", "status"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    logger.info("git working properly!")
    if prompt_type == "hybrid":
        logger.info("static update on jdk and dependency versions")
        update_jdk_related(repo_path)
        update_dependency_version(repo_path)


def load_repo_from_s3(s3_uri: str) -> str:
    s3 = boto3.client("s3")

    bucket, key = parse_s3_uri(s3_uri)

    workdir = "/tmp/workspace"

    os.makedirs(workdir, exist_ok=True)

    tar_path = os.path.join(workdir, os.path.basename(key))

    s3.download_file(bucket, key, tar_path)

    repo_id = os.path.basename(key).removesuffix(".tar.gz")
    repo_path = os.path.join(workdir, repo_id)

    # Clean up if it exists
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=workdir)

    if not os.path.exists(repo_path):
        raise ValueError(
            f"{repo_path} does not exist. only found {os.listdir(workdir)} in {workdir}. "
            "Note that tar file name should align with the compressed folder name."
        )

    os.remove(tar_path)

    # if the repo has user__repo format for its name, remove the username to simplify.
    repo_name = repo_id.split("__")[-1]
    if repo_name != repo_id:
        simple_repo_path = os.path.join(workdir, repo_name)
        if os.path.exists(simple_repo_path):
            shutil.rmtree(simple_repo_path)
        shutil.move(repo_path, simple_repo_path)
        return simple_repo_path

    return repo_path
