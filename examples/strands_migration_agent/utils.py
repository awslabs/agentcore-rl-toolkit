import json
import logging
import os
import shutil
import subprocess
import tarfile

import boto3
from java_migration_agent.preprocessing import update_dependency_version, update_jdk_related

logger = logging.getLogger(__name__)


def configure_codeartifact_token():
    domain = os.environ.get("CODEARTIFACT_DOMAIN")
    owner = os.environ.get("CODEARTIFACT_OWNER")
    repo = os.environ.get("CODEARTIFACT_REPO")
    if not domain or not owner or not repo:
        return
    ca = boto3.client("codeartifact")
    token = ca.get_authorization_token(domain=domain, domainOwner=owner)["authorizationToken"]
    url = ca.get_repository_endpoint(domain=domain, domainOwner=owner, repository=repo, format="maven")[
        "repositoryEndpoint"
    ]
    os.environ["CODEARTIFACT_AUTH_TOKEN"] = token

    m2_dir = os.path.join(os.path.expanduser("~"), ".m2")
    os.makedirs(m2_dir, exist_ok=True)
    with open(os.path.join(m2_dir, "settings.xml"), "w") as f:
        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<settings>\n"
            "  <servers>\n"
            "    <server>\n"
            "      <id>codeartifact</id>\n"
            "      <username>aws</username>\n"
            f"      <password>{token}</password>\n"
            "    </server>\n"
            "  </servers>\n"
            "  <mirrors>\n"
            "    <mirror>\n"
            "      <id>codeartifact</id>\n"
            f"      <url>{url}</url>\n"
            "      <mirrorOf>central</mirrorOf>\n"
            "    </mirror>\n"
            "  </mirrors>\n"
            "</settings>\n"
        )
    logger.info("CodeArtifact mirror configured: %s", url)


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


def setup_repo_environment(repo_path: str, apply_static_update: bool = False):
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
    if apply_static_update:
        logger.info("Apply static update on jdk and dependency versions")
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
